#!/usr/bin/env python3
"""
RAG Book Retrieval with Qwen 72B (GPTQ or bitsandbytes)
Semantic search → Top pages → Qwen 72B reads & answers
"""

# Pillow compatibility fix (must be before other imports)
from PIL import Image
if not hasattr(Image, 'LANCZOS'):
    # Pillow 10+ moved constants to Resampling enum
    if hasattr(Image, 'Resampling'):
        Image.LANCZOS = Image.Resampling.LANCZOS
        Image.BILINEAR = Image.Resampling.BILINEAR
        Image.BICUBIC = Image.Resampling.BICUBIC
        Image.NEAREST = Image.Resampling.NEAREST
        Image.BOX = Image.Resampling.BOX
        Image.HAMMING = Image.Resampling.HAMMING

import gc
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoProcessor

# Model class import
try:
    from transformers import AutoModelForImageTextToText as QwenModelClass
except Exception:
    from transformers import AutoModelForVision2Seq as QwenModelClass


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("/app")
CACHE = ROOT / "index_all_cache"

VISUAL_DIR = CACHE / "visual"
VIS_META = VISUAL_DIR / "visual_index_metadata.json"

SEM_DIR = CACHE / "semantic"
SEM_INDEX = SEM_DIR / "semantic_index.jsonl"

# Try GPTQ first (more efficient), fallback to bitsandbytes
QWEN_72B_GPTQ = "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8"
QWEN_72B_BASE = "Qwen/Qwen2-VL-72B-Instruct"

# RAG settings
TOP_K_PAGES = 10
MAX_CONTEXT_PAGES = 3  # Conservative: 3 pages = ~16k tokens (safe for 32k limit)
MAX_NEW_TOKENS = 1000  # Reduced to save memory


# ============================================================
# LOGGING
# ============================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================
# INDEX VALIDATION
# ============================================================

def validate_index() -> Tuple[bool, List[str]]:
    """Validate index exists and is readable."""
    errors = []
    
    if not VIS_META.exists():
        errors.append(f"Visual metadata missing: {VIS_META}")
    if not SEM_INDEX.exists():
        errors.append(f"Semantic index missing: {SEM_INDEX}")
    
    return (len(errors) == 0, errors)


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(s: str) -> str:
    """Normalize: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def tokenize_query(q: str) -> List[str]:
    """Extract alphanumeric tokens 3+ chars."""
    q = normalize_text(q)
    tokens = re.findall(r"[a-z0-9]+", q)
    return [t for t in tokens if len(t) >= 3]


# ============================================================
# METADATA
# ============================================================

def load_visual_metadata() -> dict:
    """Load visual metadata for book."""
    logger.info("Loading visual metadata...")
    
    with open(VIS_META, "r", encoding="utf-8") as f:
        meta_all = json.load(f)

    paper_id = next(iter(meta_all.keys()))
    meta = meta_all[paper_id]
    meta["paper_id"] = paper_id
    
    logger.info(f"  Book: {meta['pdf_filename']}")
    logger.info(f"  Pages: {meta['page_count']}")
    
    return meta


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def load_semantic_index(pdf_filename: str) -> List[dict]:
    """Load semantic records for specific PDF."""
    logger.info("Loading semantic index...")
    
    records = []
    with open(SEM_INDEX, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("pdf_filename") == pdf_filename:
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"  Loaded {len(records)} page records")
    return records


def semantic_search(
    query: str,
    records: List[dict],
    top_k: int = 50
) -> List[dict]:
    """
    Enhanced semantic search with field weighting.
    
    Scoring:
    - Exact phrase match in summary: +20 points
    - Summary token overlap: +3 points each
    - Key term overlap: +2 points each
    - Claim overlap: +1 point each
    """
    qnorm = normalize_text(query)
    qtokens = set(tokenize_query(query))
    
    if not qtokens:
        return []

    scored = []
    
    for rec in records:
        # Extract fields
        summary = normalize_text(rec.get("semantic_summary", ""))
        terms = " ".join(rec.get("key_terms", []) or [])
        claims = " ".join(rec.get("notable_claims", []) or [])
        
        terms_norm = normalize_text(terms)
        claims_norm = normalize_text(claims)
        
        # Exact phrase match in summary (high value)
        exact_in_summary = 20 if qnorm and len(qnorm) >= 4 and qnorm in summary else 0
        
        # Token overlaps with field weighting
        summary_tokens = set(re.findall(r"[a-z0-9]+", summary))
        terms_tokens = set(re.findall(r"[a-z0-9]+", terms_norm))
        claims_tokens = set(re.findall(r"[a-z0-9]+", claims_norm))
        
        summary_overlap = len(qtokens & summary_tokens) * 3
        terms_overlap = len(qtokens & terms_tokens) * 2
        claims_overlap = len(qtokens & claims_tokens) * 1
        
        score = exact_in_summary + summary_overlap + terms_overlap + claims_overlap
        
        if score > 0:
            rec2 = dict(rec)
            rec2["_search_score"] = float(score)
            rec2["_exact_match"] = int(exact_in_summary > 0)
            rec2["_summary_overlap"] = summary_overlap // 3
            rec2["_terms_overlap"] = terms_overlap // 2
            scored.append(rec2)

    scored.sort(key=lambda r: r["_search_score"], reverse=True)
    
    logger.info(f"  Semantic search: {len(scored)} hits for '{query}'")
    
    return scored[:top_k]


# ============================================================
# QWEN 72B LOADING
# ============================================================

def check_gptq_available() -> bool:
    """Check if auto-gptq is installed."""
    try:
        import auto_gptq
        return True
    except ImportError:
        return False


def load_qwen_72b_gptq() -> Tuple[any, AutoProcessor]:
    """Load GPTQ quantized model (most efficient)."""
    logger.info("Attempting to load GPTQ-Int8 model...")
    
    if not check_gptq_available():
        raise ImportError("auto-gptq not installed. Install with: pip install auto-gptq")
    
    proc = AutoProcessor.from_pretrained(QWEN_72B_GPTQ)
    
    # GPTQ loading configuration
    model = QwenModelClass.from_pretrained(
        QWEN_72B_GPTQ,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model.eval()
    logger.info("  ✅ GPTQ model loaded successfully")
    
    return model, proc


def load_qwen_72b_bitsandbytes() -> Tuple[any, AutoProcessor]:
    """Fallback: Load with bitsandbytes 8-bit quantization."""
    logger.info("Loading with bitsandbytes 8-bit quantization...")
    
    proc = AutoProcessor.from_pretrained(QWEN_72B_BASE)
    
    model = QwenModelClass.from_pretrained(
        QWEN_72B_BASE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_8bit=True
    )
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("  ✅ Gradient checkpointing enabled")
    
    model.eval()
    logger.info("  ✅ Bitsandbytes model loaded successfully")
    
    return model, proc


def load_qwen_72b() -> Tuple[any, AutoProcessor]:
    """
    Load Qwen 72B with best available quantization method.
    Tries GPTQ first (most efficient), falls back to bitsandbytes.
    """
    logger.info("Loading Qwen2-VL-72B-Instruct...")
    logger.info("  This may take 2-3 minutes...")
    
    # Try GPTQ first
    try:
        model, proc = load_qwen_72b_gptq()
    except Exception as e:
        logger.warning(f"GPTQ loading failed: {e}")
        logger.info("Falling back to bitsandbytes quantization...")
        
        try:
            model, proc = load_qwen_72b_bitsandbytes()
        except Exception as e2:
            logger.error(f"Bitsandbytes loading also failed: {e2}")
            raise RuntimeError("Failed to load model with any quantization method")
    
    # Log memory usage
    if torch.cuda.is_available():
        total_allocated = 0
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            logger.info(f"     GPU {i}: {allocated:.1f} GB allocated")
            total_allocated += allocated
        logger.info(f"     Total VRAM: {total_allocated:.1f} GB")
    
    return model, proc


# ============================================================
# IMAGE LOADING (FIXED FOR PILLOW COMPATIBILITY)
# ============================================================

def load_image_safe(image_path: str) -> Optional[Image.Image]:
    """
    Load image with Pillow compatibility handling.
    Works around Qwen processor Pillow version issues.
    """
    try:
        # Load and convert to RGB
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.warning(f"Failed to load {image_path}: {e}")
        return None


# ============================================================
# RAG SYNTHESIS
# ============================================================

def build_rag_prompt(query: str, book_title: str, total_pages: int, top_pages: List[dict]) -> str:
    """Build RAG prompt with page context."""
    
    page_context = []
    for i, page in enumerate(top_pages, 1):
        ctx = f"Page {page['page_index'] + 1}:\n"
        
        if page.get("semantic_summary"):
            ctx += f"  Summary: {page['semantic_summary']}\n"
        
        if page.get("key_terms"):
            ctx += f"  Key terms: {', '.join(page['key_terms'][:8])}\n"
        
        if page.get("notable_claims"):
            claims = page['notable_claims'][:2]
            ctx += f"  Key points: {'; '.join(claims)}\n"
        
        page_context.append(ctx)
    
    prompt = f"""You are analyzing the book "{book_title}" ({total_pages} pages total).

The user asked: "{query}"

I've identified the {len(top_pages)} most relevant pages and will show you their images. Here's what I know about them:

{chr(10).join(page_context)}

Now look at the actual page images and provide a comprehensive answer. Your response should:
1. Start with: "I searched the {book_title} ({total_pages} pages)..."
2. Synthesize information from the pages to directly answer the question
3. Reference specific pages when making claims (e.g., "Page 240 explains that...")
4. Provide technical details and examples where relevant
5. End with page numbers for further reading if appropriate

Be thorough and technical - this is for an expert reader."""

    return prompt


def synthesize_answer(
    model: any,
    proc: AutoProcessor,
    query: str,
    top_pages: List[dict],
    meta: dict
) -> str:
    """
    Use Qwen 72B to read pages and synthesize answer.
    """
    logger.info(f"Synthesizing answer from top {len(top_pages)} pages...")
    
    # Load page images with safe loading
    images = []
    loaded_pages = []
    
    for page in top_pages:
        img = load_image_safe(page["image_path"])
        if img is not None:
            images.append(img)
            loaded_pages.append(page)
    
    if not images:
        return "Error: Could not load any page images. This might be a Pillow version issue. Try: pip install Pillow --upgrade --break-system-packages"
    
    logger.info(f"  Loaded {len(images)}/{len(top_pages)} page images successfully")
    
    # Build prompt
    prompt_text = build_rag_prompt(
        query,
        meta["pdf_filename"],
        meta["page_count"],
        loaded_pages  # Use only successfully loaded pages
    )
    
    # Construct messages with multiple images
    content = []
    for _ in images:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt_text})
    
    messages = [{"role": "user", "content": content}]
    
    # Generate
    try:
        prompt = proc.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = proc(text=prompt, images=images, return_tensors="pt")
        
        # Check token length before running
        input_length = inputs["input_ids"].shape[-1]
        if input_length > 30000:  # Conservative limit (model max is 32768)
            logger.warning(f"Input tokens ({input_length}) very high. Reducing context...")
            # Retry with fewer pages
            if len(images) > 2:
                logger.info(f"  Retrying with {len(images)-1} pages instead of {len(images)}")
                return synthesize_answer(model, proc, query, loaded_pages[:-1], meta)
            else:
                return f"Error: Query context too large even with minimal pages. Try a more specific question."
        
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        
        prompt_len = inputs["input_ids"].shape[-1]
        
        logger.info(f"  Input tokens: {prompt_len}, generating {MAX_NEW_TOKENS} tokens...")
        logger.info("  This may take 30-60 seconds...")
        
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode only generated part
        gen_ids = out[0][prompt_len:]
        answer = proc.decode(gen_ids, skip_special_tokens=True).strip()
        
        logger.info("  ✅ Answer generated")
        
        # CRITICAL: Clean up tensors to free GPU memory
        del inputs
        del out
        del gen_ids
        # Also clean up images
        for img in images:
            img.close()
        del images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return answer
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"Out of memory during generation")
        
        # Clean up before retry
        if 'inputs' in locals():
            del inputs
        if 'out' in locals():
            del out
        for img in images:
            img.close()
        del images
        gc.collect()
        torch.cuda.empty_cache()
        
        # Retry with fewer pages if possible
        if len(loaded_pages) > 1:
            logger.info(f"  Retrying with {len(loaded_pages)-1} pages...")
            return synthesize_answer(model, proc, query, loaded_pages[:-1], meta)
        else:
            return (
                "Error: Out of GPU memory even with 1 page. Possible fixes:\n"
                "1. Restart the script to clear VRAM\n"
                "2. Use GPTQ model (saves ~15GB): pip install auto-gptq\n"
                "3. Try a shorter, more specific question"
            )
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        
        # Clean up on any error
        if 'inputs' in locals():
            del inputs
        if 'images' in locals():
            for img in images:
                try:
                    img.close()
                except:
                    pass
            del images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return f"Error during answer generation: {str(e)}"


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode(semantic_records: List[dict], meta: dict):
    """Interactive RAG Q&A loop."""
    
    print("\n" + "=" * 100)
    print("📚 RAG BOOK Q&A SYSTEM")
    print("=" * 100)
    print(f"Book: {meta['pdf_filename']}")
    print(f"Pages: {meta['page_count']}")
    print(f"Indexed: {len(semantic_records)} pages")
    print("\nAsk questions about the book content. Type 'exit' to quit.")
    print("=" * 100)
    
    # Load Qwen 72B once
    try:
        qwen_model, qwen_proc = load_qwen_72b()
    except Exception as e:
        print(f"\n❌ Failed to load Qwen 72B: {e}")
        print("\nPossible fixes:")
        print("  1. Install auto-gptq: pip install auto-gptq --break-system-packages")
        print("  2. Check VRAM availability (need ~36-50GB)")
        print("  3. Ensure model is downloaded in HuggingFace cache\n")
        return
    
    while True:
        try:
            query = input("\n❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        
        print("\n" + "=" * 100)
        
        # Stage 1: Semantic search
        top_pages = semantic_search(query, semantic_records, top_k=TOP_K_PAGES)
        
        if not top_pages:
            print("❌ No relevant pages found for this query.")
            print("Try rephrasing or using different keywords.\n")
            continue
        
        # Limit to avoid OOM
        context_pages = top_pages[:MAX_CONTEXT_PAGES]
        
        print(f"\n🔍 Found {len(top_pages)} relevant pages")
        print(f"📖 Reading top {len(context_pages)} pages: ", end="")
        print(", ".join([str(p["page_index"] + 1) for p in context_pages]))
        print("\n" + "-" * 100 + "\n")
        
        # Stage 2: Qwen 72B synthesis
        answer = synthesize_answer(
            qwen_model,
            qwen_proc,
            query,
            context_pages,
            meta
        )
        
        print(answer)
        print("\n" + "=" * 100)
        
        # CRITICAL: Clean up GPU memory after each query
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Report memory status
            allocated = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"GPU memory after cleanup: {allocated:.1f} GB")


# ============================================================
# MAIN
# ============================================================

def main():
    """Main entry point."""
    
    print("\n" + "=" * 100)
    print("INITIALIZING RAG SYSTEM")
    print("=" * 100)
    
    # Validate
    is_valid, errors = validate_index()
    if not is_valid:
        print("\n❌ INDEX VALIDATION FAILED:")
        for error in errors:
            print(f"  • {error}")
        print("\nRun index_all.py first.\n")
        return 1
    
    logger.info("✅ Index validation passed")
    
    # Load components
    try:
        meta = load_visual_metadata()
        semantic_records = load_semantic_index(meta["pdf_filename"])
        
        # Enter interactive mode
        interactive_mode(semantic_records, meta)
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
