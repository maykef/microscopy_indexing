#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor

# ----------------------------
# Paths (same cache layout as index_all.py)
# ----------------------------
ROOT = Path("/app")
CACHE = ROOT / "index_all_cache"

VISUAL_DIR = CACHE / "visual"
VIS_META = VISUAL_DIR / "visual_index_metadata.json"

SEM_DIR = CACHE / "semantic"
SEM_INDEX = SEM_DIR / "semantic_index.jsonl"

COLPALI_MODEL = "vidore/colpali-v1.2"


# ----------------------------
# Small text helpers
# ----------------------------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def tokenize_query(q: str) -> List[str]:
    q = normalize_text(q)
    toks = re.findall(r"[a-z0-9]+", q)
    return [t for t in toks if len(t) >= 3]


# ----------------------------
# Load metadata + embeddings
# ----------------------------
def load_visual_metadata_single_book() -> dict:
    if not VIS_META.exists():
        raise FileNotFoundError(f"Missing visual metadata: {VIS_META}")

    with open(VIS_META, "r", encoding="utf-8") as f:
        meta_all = json.load(f)

    if not meta_all:
        raise RuntimeError(f"Visual metadata file is empty: {VIS_META}")

    if len(meta_all) != 1:
        print(f"⚠️ Expected 1 book, found {len(meta_all)} entries in metadata. Using the first.")

    paper_id = next(iter(meta_all.keys()))
    meta = meta_all[paper_id]
    meta["paper_id"] = paper_id
    return meta


def load_page_embeddings(emb_path: Path) -> torch.Tensor:
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing ColPali embeddings: {emb_path}")

    page_embs = torch.load(emb_path, map_location="cpu")

    # If older format stored a list of tensors, stack/concat safely
    if isinstance(page_embs, list):
        if len(page_embs) == 0:
            raise RuntimeError(f"Embeddings file is empty list: {emb_path}")
        page_embs = torch.cat(page_embs, dim=0)

    return page_embs


# ----------------------------
# Semantic search over JSONL (cheap lexical-ish)
# ----------------------------
def semantic_search(query: str, jsonl_path: Path, top_k: int = 80) -> List[dict]:
    """
    Scores each page record by term overlap against:
      - key_terms
      - semantic_summary
      - notable_claims
    Returns top_k records with _semantic_score.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing semantic index: {jsonl_path}")

    qnorm = normalize_text(query)
    qtokens = set(tokenize_query(query))

    scored = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            kt = " ".join(rec.get("key_terms", []) or [])
            summ = rec.get("semantic_summary", "") or ""
            claims = " ".join(rec.get("notable_claims", []) or [])
            blob = normalize_text(" ".join([kt, summ, claims]))

            # Score: strong bonus for exact query substring + token overlap
            exact = 1 if qnorm and len(qnorm) >= 4 and qnorm in blob else 0
            btoks = set(re.findall(r"[a-z0-9]+", blob))
            overlap = len(qtokens.intersection(btoks))

            score = exact * 10 + overlap
            if score > 0:
                rec2 = dict(rec)
                rec2["_semantic_score"] = float(score)
                rec2["_semantic_exact"] = int(exact)
                rec2["_semantic_overlap"] = int(overlap)
                scored.append(rec2)

    scored.sort(key=lambda r: r["_semantic_score"], reverse=True)
    return scored[:top_k]


# ----------------------------
# ColPali (used as reranker)
# ----------------------------
def load_colpali():
    print("🚀 Loading ColPali...")
    try:
        model = ColPali.from_pretrained(
            COLPALI_MODEL,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
            local_files_only=True
        )
        proc = ColPaliProcessor.from_pretrained(COLPALI_MODEL, local_files_only=True)
        print("   ✅ Loaded from local cache")
    except Exception:
        model = ColPali.from_pretrained(
            COLPALI_MODEL,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2"
        )
        proc = ColPaliProcessor.from_pretrained(COLPALI_MODEL)
        print("   ✅ Downloaded and cached")
    model.eval()
    return model, proc


def colpali_query_embedding(model, proc, query: str):
    with torch.no_grad():
        inputs = proc.process_queries([query]).to(model.device)
        q_emb = model(**inputs)
    return q_emb


def score_with_colpali(model, q_emb, page_embs: torch.Tensor) -> torch.Tensor:
    """
    Preferred: use model.score if available (late-interaction scoring).
    Fallback: mean-pool cosine (less accurate).
    Returns tensor of scores length = num_pages.
    """
    if hasattr(model, "score"):
        with torch.no_grad():
            scores = model.score(q_emb, page_embs.to(model.device))
        return scores.detach().float().cpu()

    # Fallback
    q = q_emb.detach().float().cpu()
    p = page_embs.detach().float().cpu()
    qv = q.mean(dim=1) if q.ndim > 2 else q.mean(dim=0, keepdim=True)
    pv = p.mean(dim=1) if p.ndim > 2 else p
    qv = torch.nn.functional.normalize(qv, dim=-1)
    pv = torch.nn.functional.normalize(pv, dim=-1)
    return (pv @ qv.squeeze(0)).float()


# ----------------------------
# Merge / rerank
# ----------------------------
def zscore(vals: List[float]) -> List[float]:
    if not vals:
        return []
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / max(1, (len(vals) - 1))
    s = v ** 0.5
    if s == 0:
        return [0.0 for _ in vals]
    return [(x - m) / s for x in vals]


def interactive_loop(meta: dict, page_embs: torch.Tensor):
    print("\n✅ Retriever ready")
    print(f"📘 Book: {meta['pdf_filename']} ({meta['page_count']} pages)")
    print("Type 'exit' to quit.\n")

    # Load ColPali once (we will use it only to rerank semantic candidates)
    colpali_model, colpali_proc = load_colpali()

    while True:
        query = input("Query: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break

        # 1) Semantic retrieve (broad)
        sem_topk = 80          # candidates from semantic index
        final_show = 20        # results to display
        rerank_topn = 40       # how many of semantic candidates to rerank with ColPali

        sem_hits = semantic_search(query, SEM_INDEX, top_k=sem_topk)

        if not sem_hits:
            print("\n⚠️ No semantic hits found (index may not contain those terms yet).")
            print("Try a different query or wait for indexing to progress further.\n")
            continue

        # Keep only the book's PDF (defensive; semantic index can be multi-pdf later)
        sem_hits = [r for r in sem_hits if r.get("pdf_filename") == meta["pdf_filename"]]

        # Candidate page indices from semantic
        cand = sem_hits[:rerank_topn]
        cand_pages = [int(r["page_index"]) for r in cand]

        # 2) ColPali rerank those candidates
        q_emb = colpali_query_embedding(colpali_model, colpali_proc, query)

        # Subset embeddings for candidates
        cand_embs = page_embs[cand_pages]
        col_scores = score_with_colpali(colpali_model, q_emb, cand_embs).tolist()

        # Fuse: z(colpali) + z(semantic)
        sem_scores = [float(r["_semantic_score"]) for r in cand]
        cz = zscore(col_scores)
        sz = zscore(sem_scores)

        # Weighting: semantic-first pipeline, ColPali as reranker => give ColPali strong influence
        alpha = 0.60  # ColPali weight in fused score over the candidate set
        fused = [alpha * c + (1 - alpha) * s for c, s in zip(cz, sz)]

        # Build fused rows
        rows = []
        for r, cscore, fscore in zip(cand, col_scores, fused):
            page_i = int(r["page_index"])
            rows.append({
                "page_index": page_i,
                "image_path": meta["image_paths"][page_i],
                "fused_score": float(fscore),
                "colpali_score": float(cscore),
                "semantic_score": float(r["_semantic_score"]),
                "semantic_summary": r.get("semantic_summary", "") or "",
                "key_terms": r.get("key_terms", []) or [],
                "notable_claims": r.get("notable_claims", []) or [],
                "_semantic_exact": r.get("_semantic_exact", 0),
                "_semantic_overlap": r.get("_semantic_overlap", 0),
            })

        rows.sort(key=lambda x: x["fused_score"], reverse=True)

        # Print
        print("\n" + "=" * 100)
        print(f"QUERY: {query}")
        print(f"Semantic candidates: {len(sem_hits)} (reranked top {len(rows)})")
        print("=" * 100)

        for rank, r in enumerate(rows[:final_show], start=1):
            kt = ", ".join(r["key_terms"][:10]) if r["key_terms"] else ""
            claim = r["notable_claims"][0] if r["notable_claims"] else ""
            summary = r["semantic_summary"]

            print(
                f"\n#{rank:02d} page={r['page_index']:04d}  fused={r['fused_score']:+.3f}  "
                f"colpali={r['colpali_score']:.3f}  sem={r['semantic_score']:.1f} "
                f"(exact={r['_semantic_exact']} overlap={r['_semantic_overlap']})"
            )
            if summary:
                print(f"   summary: {summary}")
            if kt:
                print(f"   key_terms: {kt}")
            if claim:
                print(f"   claim: {claim}")
            print(f"   image: {r['image_path']}")

        print("\nTip: open a page image to verify quickly (path shown above).\n")


def main():
    meta = load_visual_metadata_single_book()
    emb_path = Path(meta["embedding_path"])
    page_embs = load_page_embeddings(emb_path)

    interactive_loop(meta, page_embs)


if __name__ == "__main__":
    main()
