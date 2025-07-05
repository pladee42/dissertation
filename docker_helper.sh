#!/bin/bash
# Docker Helper Script - Easy access to Docker operations from main directory

DOCKER_DIR="Docker"

echo "=== Dissertation Docker Helper (agent-env) ==="
echo "Docker files are organized in: $DOCKER_DIR/"
echo ""

case "$1" in
    "verify")
        echo "Running all verification scripts..."
        cd $DOCKER_DIR
        ./stage1_verify.sh && ./stage2_verify.sh && ./stage3_verify.sh
        ;;
    "build")
        echo "Building Docker image..."
        cd $DOCKER_DIR
        ./build_and_push.sh
        ;;
    "up")
        echo "Starting local development environment..."
        cd $DOCKER_DIR
        docker compose up -d
        echo "Environment started! Access with: docker compose exec dissertation bash"
        ;;
    "down")
        echo "Stopping local development environment..."
        cd $DOCKER_DIR
        docker compose down
        ;;
    "logs")
        echo "Showing container logs..."
        cd $DOCKER_DIR
        docker compose logs -f
        ;;
    "convert")
        echo "Converting Docker to Singularity..."
        cd $DOCKER_DIR
        ./singularity_conversion.sh
        ;;
    "clean")
        echo "Cleaning up Docker resources..."
        docker system prune -f
        ;;
    *)
        echo "Usage: $0 {verify|build|up|down|logs|convert|clean}"
        echo ""
        echo "Commands:"
        echo "  verify  - Run all verification scripts"
        echo "  build   - Build Docker image"
        echo "  up      - Start local development environment"
        echo "  down    - Stop local development environment"
        echo "  logs    - Show container logs"
        echo "  convert - Convert Docker to Singularity for HPC"
        echo "  clean   - Clean up Docker resources"
        echo ""
        echo "Docker files are located in: $DOCKER_DIR/"
        echo "For detailed instructions, see: $DOCKER_DIR/README_docker.md"
        ;;
esac