#!/bin/bash
# Build and push Docker image script - 10X Engineer approach

# Configuration
IMAGE_NAME="dissertation-env"
TAG="latest"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io/pladee42}" # Change pladee42 to your Docker Hub username

echo "=== Building GLIBCXX-compatible Docker image ==="

# Build foundation image
echo "Building foundation image..."
docker build -f Dockerfile.foundation -t $IMAGE_NAME:$TAG ..

# Test GLIBCXX compatibility
echo "Testing GLIBCXX compatibility..."
docker run --rm -v $PWD:/workspace/code $IMAGE_NAME:$TAG python3 /workspace/code/test_glibcxx.py

# Tag for registry
echo "Tagging for registry: $DOCKER_REGISTRY/$IMAGE_NAME:$TAG"
docker tag $IMAGE_NAME:$TAG $DOCKER_REGISTRY/$IMAGE_NAME:$TAG

echo ""
echo "🎯 Build complete!"
echo "📋 Next steps:"
echo "   1. Set DOCKER_REGISTRY: export DOCKER_REGISTRY=your-registry"
echo "   2. Login to registry: docker login"
echo "   3. Push image: docker push $DOCKER_REGISTRY/$IMAGE_NAME:$TAG"
echo ""
echo "Image ready: $DOCKER_REGISTRY/$IMAGE_NAME:$TAG"