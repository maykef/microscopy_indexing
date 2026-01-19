#!/bin/bash

IMAGE_NAME="microscopy_index"
CONTAINER_NAME="microscopy_index"

# PATHS (same style as your original)
DATA_DIR="/mnt/nvme8tb/microscopy_index"
NVME_DIR="/mnt/nvme8tb/huggingface_cache/hub"

echo "🏗️  Building Blackwell-Optimized Image..."
docker build -t $IMAGE_NAME .

# Clean up any existing container
docker rm -f $CONTAINER_NAME 2>/dev/null

echo "🚀 Launching Index Node..."
docker run --gpus all -it \
  --name $CONTAINER_NAME \
  --ipc=host \
  --shm-size=16gb \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $DATA_DIR:/app \
  -v $NVME_DIR:/root/.cache/huggingface/hub \
  $IMAGE_NAME
