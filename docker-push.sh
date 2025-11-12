#!/bin/bash
# Push VGGT Docker image to GitHub Container Registry

set -e

IMAGE_NAME="ghcr.io/infinitespacesorg/vggt"
VERSION="${1:-latest}"

echo "Pushing VGGT Docker image to GitHub Container Registry"
echo "Image: ${IMAGE_NAME}:${VERSION}"
echo ""

# Check if logged in to ghcr.io
if ! docker info 2>/dev/null | grep -q "ghcr.io"; then
    echo "⚠ Not logged in to GitHub Container Registry"
    echo ""
    echo "Please login first:"
    echo "  echo \$GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin"
    echo ""
    echo "Or create a Personal Access Token at:"
    echo "  https://github.com/settings/tokens/new"
    echo "  (Requires 'write:packages' scope)"
    echo ""
    exit 1
fi

# Check if image exists locally
if ! docker image inspect ${IMAGE_NAME}:${VERSION} &> /dev/null; then
    echo "❌ Image ${IMAGE_NAME}:${VERSION} not found locally"
    echo ""
    echo "Build it first:"
    echo "  ./docker-build.sh ${VERSION}"
    echo ""
    exit 1
fi

echo "Pushing ${IMAGE_NAME}:${VERSION}..."
docker push ${IMAGE_NAME}:${VERSION}

echo ""
echo "✓ Push complete!"
echo ""
echo "Image available at:"
echo "  ${IMAGE_NAME}:${VERSION}"
echo ""
echo "Pull with:"
echo "  docker pull ${IMAGE_NAME}:${VERSION}"

