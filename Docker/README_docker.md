# Container Deployment Guide

This README provides step-by-step instructions for deploying your dissertation environment across different platforms using the containerized setup.

**📂 File Organization:** All Docker-related files are organized in the `Docker/` folder for better project structure.

## Quick Start

```bash
# From main project directory
cd Docker

# Verify all stages are complete
./stage1_verify.sh && ./stage2_verify.sh && ./stage3_verify.sh

# Set your Docker registry
export DOCKER_REGISTRY=your-dockerhub-username

# Build and test locally
./build_and_push.sh
```

**Alternative:** Use the helper script from the main directory:
```bash
# From main project directory
chmod +x docker_helper.sh
./docker_helper.sh verify  # Verify all stages
./docker_helper.sh build   # Build image
./docker_helper.sh up      # Start local environment
```

## Platform Deployment Options

### 🌩️ Cloud Deployment (Vast.ai)

**Best for:** Fast iteration, GPU experimentation, cost-effective training

#### Step 1: Prepare Image
```bash
# Build and test image
./build_and_push.sh

# Push to Docker registry
docker login
docker push $DOCKER_REGISTRY/dissertation-env:latest
```

#### Step 2: Deploy on Vast.ai
1. Go to [Vast.ai Console](https://cloud.vast.ai/)
2. Search for instances with required specs:
   - **GPU**: Based on your model requirements (8GB+ recommended)
   - **vRAM**: See model requirements in [config/config.py](config/config.py)
   - **Disk**: 100GB+ for model cache
3. Use custom image: `your-dockerhub-username/dissertation-env:latest`
4. Set environment variables:
   ```
   PYTHONUNBUFFERED=1
   CUDA_DEVICE_MAX_CONNECTIONS=1
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   HF_HOME=/workspace/downloaded_models
   ```
5. Add volumes:
   ```
   model_cache:/workspace/downloaded_models
   project_outputs:/workspace/output
   project_logs:/workspace/log
   ```
6. Set onstart command: `bash /workspace/code/vast_setup.sh`

#### Step 3: Connect and Run
```bash
# SSH to instance
ssh -p <port> root@<instance-ip>

# Your code is in /workspace/code
cd /workspace/code

# Run your experiments
python -m runner --email_generation=medium --topic="Your Topic"
```

---

### 🎓 HPC Deployment (Sheffield)

**Best for:** Large-scale experiments, institutional computing, long-running jobs

#### Step 1: Convert to Singularity
```bash
# On local machine with Singularity installed
export DOCKER_REGISTRY=your-dockerhub-username
./singularity_conversion.sh

# This creates: dissertation-env.sif
```

#### Step 2: Transfer to Sheffield HPC
```bash
# Transfer Singularity image
scp dissertation-env.sif username@sharc.sheffield.ac.uk:~/

# Transfer project files (excluding model cache)
rsync -av --exclude='downloaded_models' ./dissertation/ username@sharc.sheffield.ac.uk:~/dissertation/
```

#### Step 3: Submit SLURM Job
```bash
# SSH to Sheffield HPC
ssh username@sharc.sheffield.ac.uk

# Navigate to project
cd dissertation

# Submit containerized job
sbatch slurm_container_job.sh

# Monitor job
squeue -u $USER
watch squeue -u $USER

# View output
tail -f container_job_*.out
```

#### Customizing SLURM Jobs
Edit `slurm_container_job.sh` for your needs:
```bash
#SBATCH --mem=64G          # Increase memory
#SBATCH --time=24:00:00    # Longer runtime
#SBATCH --gres=gpu:2       # Multiple GPUs
```

---

### 💻 Local Development

**Best for:** Development, debugging, testing

#### Quick Start
```bash
# Start local environment
docker-compose up -d

# Access container
docker-compose exec dissertation bash

# Run experiments
python -m runner --email_generation=small --topic="Test Topic"
```

#### Development Workflow
```bash
# Build and test
./build_and_push.sh

# Start development environment
docker-compose up -d

# View logs
docker-compose logs -f

# Stop environment
docker-compose down
```

---

## Platform Comparison

| Feature | Local | Vast.ai | Sheffield HPC |
|---------|-------|---------|---------------|
| **Setup Time** | Instant | 2-5 minutes | 10-30 minutes |
| **Cost** | Free (your hardware) | $0.50-2.00/hour | Free (institutional) |
| **GPU Access** | Limited | High-end GPUs | Institutional GPUs |
| **Storage** | Limited | 100GB-1TB | Shared filesystem |
| **Best For** | Development | Experiments | Production runs |

---

## Troubleshooting

### Common Issues

#### 1. Docker Build Fails
```bash
# Check Docker daemon
docker version

# Clean build cache
docker system prune -f

# Rebuild from scratch
./build_and_push.sh
```

#### 2. GLIBCXX Errors (Should be eliminated!)
```bash
# Test compatibility
docker run --rm your-registry/dissertation-env:latest python3 /workspace/test_glibcxx.py

# Check glibc version
docker run --rm your-registry/dissertation-env:latest ldd --version
```

#### 3. GPU Not Accessible
```bash
# Vast.ai: Check instance has GPU
nvidia-smi

# HPC: Ensure --nv flag is used
singularity exec --nv dissertation-env.sif nvidia-smi

# Local: Install nvidia-docker2
```

#### 4. Model Download Issues
```bash
# Check HF_HOME is set
echo $HF_HOME

# Test internet connectivity
wget -q --spider https://huggingface.co

# Clear cache if needed
rm -rf /workspace/downloaded_models/*
```

### Debug Commands

```bash
# Test environment inside container
python3 test_glibcxx.py

# Check available libraries
python3 -c "import torch, vllm, transformers; print('All imports successful')"

# Monitor GPU usage
nvidia-smi -l 1

# Check disk space
df -h /workspace/downloaded_models
```

---

## Performance Optimization

### Model Caching
- **Vast.ai**: Use persistent volumes to cache models between instances
- **HPC**: Store models in shared filesystem or personal space
- **Local**: Docker volumes automatically persist models

### Memory Management
```bash
# Monitor memory usage
htop

# Clear Python cache
python3 -c "import gc; gc.collect()"

# Restart container if needed
docker-compose restart
```

### GPU Optimization
```bash
# Check GPU utilization
nvidia-smi

# Monitor GPU memory
watch -n 1 nvidia-smi

# Set memory growth (already configured)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Cost Optimization

### Vast.ai Tips
- Use spot instances for non-critical work
- Stop instances when not in use
- Choose regions with lower prices
- Monitor spending regularly

### Sheffield HPC Best Practices
- Request appropriate resources (don't over-allocate)
- Use job arrays for multiple experiments
- Clean up output files regularly
- Follow fair usage policies

---

## Security Considerations

### Secrets Management
```bash
# Never commit API keys or credentials
echo "*.key" >> .gitignore
echo "*.env" >> .gitignore

# Use environment variables
export HUGGINGFACE_TOKEN=your-token
```

### Data Privacy
- Model cache may contain temporary files
- Clean up sensitive outputs before sharing
- Use institutional guidelines for data handling

---

## Quick Reference

### Essential Commands

```bash
# From main project directory
./docker_helper.sh verify     # Verify all stages
./docker_helper.sh build      # Build and test
./docker_helper.sh up         # Local development
./docker_helper.sh convert    # Convert to Singularity

# Or from Docker/ directory
cd Docker
./build_and_push.sh          # Build and test
./singularity_conversion.sh  # HPC conversion  
docker-compose up -d         # Local development
```

### File Structure
```
Project Root/
├── docker_helper.sh              # Helper script (main directory)
├── requirements.txt               # Dependencies (main directory)
├── Docker/                        # All Docker files organized here
│   ├── Dockerfile.foundation      # Container definition
│   ├── docker-compose.yml         # Local development
│   ├── build_and_push.sh         # Build automation
│   ├── vast_setup.sh             # Vast.ai provisioning
│   ├── vast_template_config.json # Vast.ai template
│   ├── singularity_conversion.sh # HPC conversion
│   ├── slurm_container_job.sh    # HPC job script
│   ├── hpc_setup_guide.md        # HPC detailed guide
│   ├── test_glibcxx.py           # Compatibility testing
│   ├── stage1_verify.sh          # Stage 1 verification
│   ├── stage2_verify.sh          # Stage 2 verification
│   ├── stage3_verify.sh          # Stage 3 verification
│   ├── claude_docker.md          # Implementation guide
│   └── README_docker.md          # This file
└── [other project files...]
```

This containerized approach eliminates GLIBCXX issues and provides consistent environments across all platforms. Choose the deployment option that best fits your current needs!