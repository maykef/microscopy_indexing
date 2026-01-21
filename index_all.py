#!/usr/bin/env python3
"""
index_all.py - Production RAG indexing with resume capability
- ColPali visual embeddings (unchanged)
- Full-text scribe transcription (replaces semantic extraction)
- Crash recovery: tracks progress, resumes from last processed page
"""

import json
import hashlib
from pathlib import Path

import torch
from tqdm import tqdm
from pdf2image import convert_from_path
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as QwenModelClass
except Exception:
    from transformers import AutoModelForVision2Seq as QwenModelClass


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/app")
CACHE = ROOT / "index_all_cache"

VISUAL_DIR = CACHE / "visual"
VIS_IMG_DIR = VISUAL_DIR / "images"
VIS_EMB_DIR = VISUAL_DIR / "embeddings"
VIS_META = VISUAL_DIR / "visual_index_metadata.json"

SCRIBE_DIR = CACHE / "scribe"
SCRIBE_INDEX = SCRIBE_DIR / "full_text_transcriptions.jsonl"

# ============================================================
# SETTINGS
# ============================================================

DPI = 200
PDF_RENDER_THREADS = 16
COLPALI_BATCH = 16

# Scribe settings
MAX_NEW_TOKENS = 16384  # Increased for dense pages
SCRIBE_FRACTION = 1.0  # Full book

COLPALI_MODEL = "vidore/colpali-v1.2"
QWEN7B_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

# Transcription prompt
TRANSCRIPTION_PROMPT = """You are transcribing a microscopy textbook page to plain text.

READ THE ENTIRE PAGE and transcribe ALL visible text in reading order (left to right, top to bottom).

Include:
- Page number and chapter title at top
- ALL body text (every paragraph, every sentence)
- ALL figure captions (complete text)
- ALL table captions and table content (every row, every cell)
- ALL equations (in LaTeX format if possible)
- ALL footnotes and references

Rules:
1. Transcribe VERBATIM - do not summarize or skip text
2. Preserve all technical terms, numbers, symbols, citations
3. Continue transcribing until you reach the bottom of the page
4. For tables: transcribe as plain text with clear structure (use | for columns)
5. Mark unclear text as [unclear]

Begin full transcription:"""


# ============================================================
# UTILS
# ============================================================

def ensure_dirs():
    VIS_IMG_DIR.mkdir(parents=True, exist_ok=True)
    VIS_EMB_DIR.mkdir(parents=True, exist_ok=True)
    SCRIBE_DIR.mkdir(parents=True, exist_ok=True)


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


def load_completed_pages(jsonl_path: Path) -> set:
    """
    Load set of (pdf_filename, page_index) tuples that have been completed.
    Enables resume capability.
    """
    completed = set()
    if not jsonl_path.exists():
        return completed
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                completed.add((obj.get("pdf_filename"), obj.get("page_index")))
            except Exception:
                continue
    
    return completed


# ============================================================
# PDF -> IMAGE CACHE
# ============================================================

def render_pdf(pdf_path: Path, paper_id: str):
    """
    Render PDF pages to JPEGs under:
      /app/index_all_cache/visual/images/<paper_id>/page_0000.jpg ...
    If already present, reuse.
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
# COLPALI (UNCHANGED)
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
    """Batch embedding with page-level progress."""
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
# SCRIBE MODE (REPLACES SEMANTIC EXTRACTION)
# ============================================================

def load_qwen7b():
    """Load Qwen2-VL-7B for full-text transcription."""
    print("🤖 Loading Qwen2-VL-7B for scribe mode...")
    proc = AutoProcessor.from_pretrained(QWEN7B_MODEL)
    model = QwenModelClass.from_pretrained(
        QWEN7B_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print("   ✅ Qwen2-VL-7B ready")
    return model, proc


def transcribe_page(model, proc, image: Image.Image) -> dict:
    """
    Full-text transcription of a single page.
    Returns: {text, tokens_generated, hit_token_limit}
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": TRANSCRIPTION_PROMPT}
            ]
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
            max_new_tokens=MAX_NEW_TOKENS
        )
    
    gen_ids = out[0][prompt_len:]
    text = proc.decode(gen_ids, skip_special_tokens=True).strip()
    
    tokens_generated = len(gen_ids)
    hit_limit = tokens_generated >= MAX_NEW_TOKENS - 10
    
    # Cleanup
    del inputs, out, gen_ids
    torch.cuda.empty_cache()
    
    return {
        "text": text,
        "tokens_generated": tokens_generated,
        "hit_token_limit": hit_limit
    }


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

    # Load completed pages for resume capability
    completed_pages = load_completed_pages(SCRIBE_INDEX)
    if completed_pages:
        print(f"📋 Found {len(completed_pages)} already-transcribed pages")
        print(f"   Will resume from where we left off")

    # ----------------------------
    # Phase 1: ColPali visual index (skip if done)
    # ----------------------------
    print("\n" + "="*80)
    print("PHASE 1: COLPALI VISUAL INDEXING")
    print("="*80)
    
    colpali, colpali_proc = load_colpali()

    for pdf in tqdm(pdfs, desc="📄 Visual indexing", unit="pdf"):
        paper_id = pdf.stem
        pdf_hash = sha1_file(pdf)
        emb_path = VIS_EMB_DIR / f"{paper_id}.pt"

        image_paths = render_pdf(pdf, paper_id)
        page_count = len(image_paths)

        if paper_id in visual_meta and emb_path.exists():
            if visual_meta[paper_id].get("pdf_hash") == pdf_hash and visual_meta[paper_id].get("page_count") == page_count:
                print(f"   ✓ {pdf.name}: Already indexed")
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
        print(f"   ✅ Saved embeddings to {emb_path.name}")

    # Free VRAM
    del colpali, colpali_proc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------------------
    # Phase 2: Scribe mode full-text transcription
    # ----------------------------
    print("\n" + "="*80)
    print("PHASE 2: FULL-TEXT TRANSCRIPTION (SCRIBE MODE)")
    print("="*80)
    print(f"Token limit: {MAX_NEW_TOKENS}")
    print(f"Resume capability: {'✅ ENABLED' if completed_pages else '✅ ENABLED (fresh start)'}")
    
    qwen, qwen_proc = load_qwen7b()

    # Open JSONL in append mode for incremental writing
    with open(SCRIBE_INDEX, "a", encoding="utf-8") as out:
        for pdf in tqdm(pdfs, desc="📄 Transcribing", unit="pdf"):
            paper_id = pdf.stem
            meta = visual_meta.get(paper_id) or {}
            image_paths = meta.get("image_paths") or render_pdf(pdf, paper_id)
            page_count = meta.get("page_count", len(image_paths))

            target_n = max(1, int(page_count * SCRIBE_FRACTION))
            target_paths = image_paths[:target_n]

            print(f"\n📗 Transcribing: {pdf.name} ({target_n}/{page_count} pages)")
            
            # Track stats for this PDF
            pages_skipped = 0
            pages_processed = 0
            token_limit_hits = 0

            for page_idx, img_path in enumerate(
                tqdm(target_paths, desc=f"🖋️ Scribe ({pdf.name})", unit="page")
            ):
                # Resume check: skip if already done
                key = (pdf.name, page_idx)
                if key in completed_pages:
                    pages_skipped += 1
                    continue

                try:
                    img = Image.open(img_path).convert("RGB")
                    result = transcribe_page(qwen, qwen_proc, img)
                    img.close()

                    # Create record
                    rec = {
                        "pdf_filename": pdf.name,
                        "paper_id": paper_id,
                        "page_index": page_idx,
                        "image_path": img_path,
                        "full_text": result["text"],
                        "tokens_generated": result["tokens_generated"],
                        "hit_token_limit": result["hit_token_limit"]
                    }

                    # Write immediately (crash-safe)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()

                    # Update local tracking
                    completed_pages.add(key)
                    pages_processed += 1
                    
                    if result["hit_token_limit"]:
                        token_limit_hits += 1

                except Exception as e:
                    print(f"\n   ⚠️ Error on page {page_idx}: {e}")
                    # Continue to next page (don't crash entire run)
                    continue

            # Summary for this PDF
            if pages_skipped > 0:
                print(f"   ↻ Skipped {pages_skipped} already-completed pages")
            if pages_processed > 0:
                print(f"   ✅ Transcribed {pages_processed} new pages")
            if token_limit_hits > 0:
                print(f"   ⚠️  {token_limit_hits} pages hit token limit (may be incomplete)")

    print("\n" + "="*80)
    print("✅ FULL INDEXING COMPLETE")
    print("="*80)
    print(f"📦 Visual embeddings: {VIS_META}")
    print(f"📝 Full-text transcriptions: {SCRIBE_INDEX}")
    print(f"   Total pages in index: {len(completed_pages)}")
    print("="*80)


if __name__ == "__main__":
    main()
