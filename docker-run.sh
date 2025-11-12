#!/bin/bash
# Run the VGGT Docker container

set -e

# Use local tag by default, or specify image name as first argument
IMAGE_NAME="${1:-vggt:latest}"

echo "Starting VGGT Docker container: ${IMAGE_NAME}"

# Check if nvidia-docker is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    GPU_FLAGS="--gpus all"
else
    echo "⚠ No NVIDIA GPU detected - running in CPU mode"
    GPU_FLAGS=""
fi

# Run the container
docker run -it --rm \
    $GPU_FLAGS \
    -p 7860:7860 \
    -v $(pwd)/outputs:/app/outputs \
    --name vggt-demo \
    ${IMAGE_NAME}

echo ""
echo "Container stopped."

