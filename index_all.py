#!/usr/bin/env python3
import json
import hashlib
from pathlib import Path

import torch
from tqdm import tqdm
from pdf2image import convert_from_path
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import AutoProcessor

# Prefer new class if present; fallback otherwise
try:
    from transformers import AutoModelForImageTextToText as QwenModelClass
except Exception:
    from transformers import AutoModelForVision2Seq as QwenModelClass

# ============================================================
# PATHS (container: /app)
# ============================================================

ROOT = Path("/app")
CACHE = ROOT / "index_all_cache"

VISUAL_DIR = CACHE / "visual"
VIS_IMG_DIR = VISUAL_DIR / "images"
VIS_EMB_DIR = VISUAL_DIR / "embeddings"
VIS_META = VISUAL_DIR / "visual_index_metadata.json"

SEM_DIR = CACHE / "semantic"
# FULL RUN OUTPUT (new clean file)
SEM_INDEX_V2 = SEM_DIR / "semantic_index_v2.jsonl"

# ============================================================
# SETTINGS
# ============================================================

DPI = 200
PDF_RENDER_THREADS = 16
COLPALI_BATCH = 16

# Full book
SEMANTIC_FRACTION = 1.0

COLPALI_MODEL = "vidore/colpali-v1.2"
QWEN7B_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

# ============================================================
# UTILS
# ============================================================

def ensure_dirs():
    VIS_IMG_DIR.mkdir(parents=True, exist_ok=True)
    VIS_EMB_DIR.mkdir(parents=True, exist_ok=True)
    SEM_DIR.mkdir(parents=True, exist_ok=True)

def list_pdfs():
    return sorted([p for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])

def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def load_json(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_done_semantic(jsonl_path: Path):
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add((obj.get("pdf_filename"), obj.get("page_index")))
            except Exception:
                continue
    return done

# ============================================================
# PDF -> IMAGE CACHE
# ============================================================

def render_pdf(pdf_path: Path, paper_id: str):
    """
    Render PDF pages to JPEGs under:
      /app/index_all_cache/visual/images/<paper_id>/page_0000.jpg ...
    If already present, reuse.
    Returns list of image paths in order.
    """
    out_dir = VIS_IMG_DIR / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("page_*.jpg"))
    if existing:
        return [str(p) for p in existing]

    pages = convert_from_path(
        str(pdf_path),
        dpi=DPI,
        thread_count=PDF_RENDER_THREADS
    )

    paths = []
    for i, page in enumerate(pages):
        p = out_dir / f"page_{i:04d}.jpg"
        page.save(p, "JPEG", quality=95, optimize=True)
        paths.append(str(p))
    return paths

# ============================================================
# COLPALI (unchanged; will skip if already indexed)
# ============================================================

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

def embed_images_colpali(model, proc, image_paths, desc):
    """
    Batches for speed, reports PAGE progress, loads images per batch (low RAM).
    """
    total_pages = len(image_paths)
    chunks = []
    with tqdm(total=total_pages, desc=desc, unit="page") as pbar:
        for i in range(0, total_pages, COLPALI_BATCH):
            batch_paths = image_paths[i:i + COLPALI_BATCH]
            batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            with torch.no_grad():
                inputs = proc.process_images(batch_imgs).to(model.device)
                out = model(**inputs)
            chunks.append(out.cpu())
            pbar.update(len(batch_imgs))
    return torch.cat(chunks, dim=0)

# ============================================================
# QWEN 7B (FIXED OUTPUT DECODING)
# ============================================================

def load_qwen7b():
    print("🤖 Loading Qwen2-VL-7B...")
    proc = AutoProcessor.from_pretrained(QWEN7B_MODEL)
    model = QwenModelClass.from_pretrained(
        QWEN7B_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print("   ✅ Qwen2-VL-7B ready")
    return model, proc

def extract_json_block(txt: str):
    txt = (txt or "").strip()
    l = txt.find("{")
    r = txt.rfind("}")
    if l != -1 and r != -1 and r > l:
        sub = txt[l:r+1]
        try:
            return json.loads(sub), txt
        except Exception:
            return None, txt
    return None, txt

def qwen_index_page(model, proc, image: Image.Image):
    """
    Correct Qwen2-VL usage:
    - Use chat template with explicit image slot
    - Decode ONLY newly generated tokens (prevents saving the prompt)
    """
    instruction = (
        "You are indexing a scientific book page.\n"
        "Return ONLY valid JSON (no markdown, no extra text) with EXACT keys:\n"
        '{\n'
        '  "semantic_summary": "1-2 sentences",\n'
        '  "key_terms": ["term", "..."],\n'
        '  "notable_claims": ["claim", "..."]\n'
        '}\n'
        "Rules:\n"
        "- If nothing useful, return empty summary and empty lists.\n"
        "- key_terms should include technical terms you can SEE on the page.\n"
        "- notable_claims should be concise factual statements from the page.\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=prompt, images=[image], return_tensors="pt")

    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=False
        )

    # Decode only generated completion
    gen_ids = out[0][prompt_len:]
    txt = proc.decode(gen_ids, skip_special_tokens=True).strip()

    parsed, raw = extract_json_block(txt)
    if parsed is None:
        parsed = {"semantic_summary": "", "key_terms": [], "notable_claims": []}
    return parsed, txt

# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dirs()

    pdfs = list_pdfs()
    if not pdfs:
        print("❌ No PDFs found in /app")
        return

    print(f"📚 Found {len(pdfs)} PDF(s)")
    visual_meta = load_json(VIS_META, {})

    # ----------------------------
    # Phase 1: ColPali visual index (skip if already done)
    # ----------------------------
    colpali, colpali_proc = load_colpali()

    for pdf in tqdm(pdfs, desc="📄 Visual indexing", unit="pdf"):
        paper_id = pdf.stem
        pdf_hash = sha1_file(pdf)
        emb_path = VIS_EMB_DIR / f"{paper_id}.pt"

        image_paths = render_pdf(pdf, paper_id)
        page_count = len(image_paths)

        if paper_id in visual_meta and emb_path.exists():
            if visual_meta[paper_id].get("pdf_hash") == pdf_hash and visual_meta[paper_id].get("page_count") == page_count:
                continue

        print(f"\n📘 ColPali indexing: {pdf.name} ({page_count} pages)")
        emb = embed_images_colpali(
            colpali,
            colpali_proc,
            image_paths,
            desc=f"👁️ ColPali pages ({pdf.name})"
        )
        torch.save(emb, emb_path)

        visual_meta[paper_id] = {
            "pdf_filename": pdf.name,
            "pdf_hash": pdf_hash,
            "page_count": page_count,
            "image_paths": image_paths,
            "embedding_path": str(emb_path),
            "dpi": DPI
        }
        save_json(VIS_META, visual_meta)

    # Free VRAM before Qwen
    del colpali, colpali_proc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------------------
    # Phase 2: Qwen7B semantic index (FULL BOOK)
    # ----------------------------
    qwen, qwen_proc = load_qwen7b()
    done = load_done_semantic(SEM_INDEX_V2)

    with open(SEM_INDEX_V2, "a", encoding="utf-8") as out:
        for pdf in tqdm(pdfs, desc="📄 Semantic indexing", unit="pdf"):
            paper_id = pdf.stem
            meta = visual_meta.get(paper_id) or {}
            image_paths = meta.get("image_paths") or render_pdf(pdf, paper_id)
            page_count = meta.get("page_count", len(image_paths))

            target_n = max(1, int(page_count * SEMANTIC_FRACTION))
            target_paths = image_paths[:target_n]

            print(f"\n📗 Qwen7B indexing FULL: {pdf.name} ({target_n}/{page_count} pages) -> {SEM_INDEX_V2.name}")

            for page_idx, img_path in enumerate(
                tqdm(target_paths, desc=f"🧠 Qwen7B pages ({pdf.name})", unit="page")
            ):
                key = (pdf.name, page_idx)
                if key in done:
                    continue

                img = Image.open(img_path).convert("RGB")
                parsed, raw = qwen_index_page(qwen, qwen_proc, img)

                rec = {
                    "pdf_filename": pdf.name,
                    "paper_id": paper_id,
                    "page_index": page_idx,
                    "image_path": img_path,
                    "semantic_summary": parsed.get("semantic_summary", ""),
                    "key_terms": parsed.get("key_terms", []),
                    "notable_claims": parsed.get("notable_claims", []),
                    "raw_output": raw
                }

                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                done.add(key)

    print("\n✅ Full semantic indexing complete")
    print(f"📦 Wrote: {SEM_INDEX_V2}")

if __name__ == "__main__":
    main()
