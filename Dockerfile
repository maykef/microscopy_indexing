# Base image (same)
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

# Env (same)
ENV DEBIAN_FRONTEND=interactive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (same)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    nano \
    git \
    poppler-utils \
    libgl1-mesa-glx \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# pip upgrade (same)
RUN pip3 install --upgrade pip

# PyTorch nightly (same)
RUN pip3 install --no-cache-dir --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu126

# Build helpers (same)
RUN pip3 install packaging setuptools wheel

# Core deps (same list, includes tqdm and protobuf)
RUN pip3 install --no-cache-dir \
    colpali-engine \
    transformers \
    accelerate \
    einops \
    bitsandbytes \
    pdf2image \
    qwen-vl-utils \
    sentencepiece \
    pillow \
    tqdm \
    protobuf

# Flash Attention (same)
RUN pip3 install flash-attn --no-build-isolation

# Pre-download ColPali (same)
RUN python3 -c "from colpali_engine.models import ColPali, ColPaliProcessor; \
    print('Downloading ColPali...'); \
    ColPali.from_pretrained('vidore/colpali-v1.2', torch_dtype='float32', device_map='cpu'); \
    ColPaliProcessor.from_pretrained('vidore/colpali-v1.2'); \
    print('✅ ColPali cached')"

# Hardware check script (same)
RUN echo 'import torch; \
import flash_attn; \
import qwen_vl_utils; \
name = torch.cuda.get_device_name(0); \
vram = torch.cuda.get_device_properties(0).total_memory / 1e9; \
print("-" * 40); \
print(f"✅ DEVICE: {name}"); \
print(f"✅ VRAM: {vram:.1f} GB"); \
print(f"✅ FLASH ATTENTION: LOADED"); \
print(f"✅ QWEN UTILS: LOADED"); \
print("-" * 40)' > check_hw.py

# ONLY CHANGE from your original container behavior:
# stay interactive by default
CMD ["/bin/bash"]
