# Sheffield HPC Container Setup Guide

## Prerequisites

1. **Access to Sheffield HPC** with SLURM
2. **Singularity** available (load with `module load Singularity`)
3. **Built Docker image** from Stage 2

## Setup Steps

### Step 1: Convert Docker to Singularity
```bash
# On your local machine (with Singularity installed)
export DOCKER_REGISTRY=your-registry
./singularity_conversion.sh

# Transfer to Sheffield HPC
scp dissertation-env.sif username@sharc.sheffield.ac.uk:~/
```

### Step 2: Transfer Project Files
```bash
# Copy your project files (excluding models cache)
rsync -av --exclude='downloaded_models' ./dissertation/ username@sharc.sheffield.ac.uk:~/dissertation/
```

### Step 3: Test on Sheffield HPC
```bash
# Login to Sheffield HPC
ssh username@sharc.sheffield.ac.uk

# Load Singularity module
module load Singularity

# Test image
cd dissertation
singularity exec --nv dissertation-env.sif python3 -c "import vllm; print('Success')"
```

### Step 4: Submit SLURM Job
```bash
# Submit containerized job
sbatch slurm_container_job.sh

# Check job status
squeue -u $USER

# View output
tail -f container_job_*.out
```

## Benefits on Sheffield HPC

✅ **GLIBCXX Issues Resolved** - Modern libraries bundled in container  
✅ **Consistent Environment** - Same setup as Vast.ai and local  
✅ **No Module Conflicts** - Isolated from system libraries  
✅ **Reproducible Results** - Identical environment across platforms

## Troubleshooting

### Common Issues:
1. **Module not found**: Load required modules in SLURM script
2. **GPU not accessible**: Use `--nv` flag with Singularity  
3. **Permission errors**: Check file permissions and bind mounts
4. **Memory issues**: Adjust SLURM memory allocation

### Debug Commands:
```bash
# Check available modules
module avail

# Test GPU in container
singularity exec --nv dissertation-env.sif nvidia-smi

# Interactive debugging
srun --pty singularity shell --nv dissertation-env.sif
```

## Quick Reference

```bash
# Convert Docker to Singularity
./singularity_conversion.sh

# Submit job
sbatch slurm_container_job.sh

# Check status
squeue -u $USER

# Cancel job
scancel JOBID
```

This containerized approach eliminates GLIBCXX compatibility issues while providing consistent performance across all platforms.