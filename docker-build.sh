#!/bin/bash
# Build the VGGT Docker image

set -e

IMAGE_NAME="ghcr.io/infinitespacesorg/vggt"
VERSION="${1:-latest}"

echo "Building VGGT Docker image: ${IMAGE_NAME}:${VERSION}"
echo "This will take several minutes as it:"
echo "  1. Sets up the Pixi environment"
echo "  2. Pre-downloads the 4.68GB VGGT model"
echo ""

# Build with both ghcr.io tag and local convenience tag
docker build -t ${IMAGE_NAME}:${VERSION} -t vggt:latest .

echo ""
echo "✓ Build complete!"
echo ""
echo "Image tagged as:"
echo "  ${IMAGE_NAME}:${VERSION}"
echo "  vggt:latest (local convenience tag)"
echo ""
echo "To run the container:"
echo "  ./docker-run.sh"
echo ""
echo "Or with Docker Compose:"
echo "  docker-compose up"
echo ""
echo "To push to GitHub Container Registry:"
echo "  docker push ${IMAGE_NAME}:${VERSION}"

