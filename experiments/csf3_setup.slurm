#!/bin/bash --login
#===============================================================================
# CSF3 Environment Setup Job Script
# Creates conda environment with TensorFlow GPU for experiment
# Run ONCE before submitting the experiment job
#===============================================================================

### GPU Partition - A100 needed to verify GPU detection during setup
#SBATCH -p gpuA              # A100 (80GB) GPUs [up to 12 CPU cores per GPU]

### Resources
#SBATCH -G 1                 # 1 GPU for TF GPU verification
#SBATCH -n 1                 # 1 CPU core (setup doesn't need more)
#SBATCH -t 0-1               # 1 hour wallclock (pip install takes ~15 min)
#SBATCH --mem=8G             # 8GB RAM for installation

### Job identification
#SBATCH -J csf3_setup        # Job name
#SBATCH -o csf3_setup_%j.out # Standard output log
#SBATCH -e csf3_setup_%j.err # Standard error log

#===============================================================================
# Environment Setup
#===============================================================================

set -e

module purge
module load libs/cuda
module load apps/binapps/conda/miniforge3/25.9.1

echo "=============================================="
echo "CSF3 Experiment Environment Setup"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

#===============================================================================
# Create Conda Environment
#===============================================================================

echo ""
echo "Creating conda environment 'experiment' with Python 3.12..."
conda create -n traffic_pred python=3.12 -y

conda activate traffic_pred

echo "Installing TensorFlow with GPU support..."
pip install --no-cache-dir "tensorflow[and-cuda]"

echo "Installing experiment dependencies..."
pip install --no-cache-dir pandas pyarrow scikit-learn geopy psutil

#===============================================================================
# Verify GPU Detection
#===============================================================================

echo ""
echo "=============================================="
echo "Verifying GPU Detection"
echo "=============================================="

python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'TensorFlow GPUs detected: {len(gpus)}')
for g in gpus:
    print(f'  - {g.name}')
if not gpus:
    print('WARNING: No GPU detected!')
else:
    print('GPU setup OK')
"

#===============================================================================
# Completion
#===============================================================================

echo ""
echo "=============================================="
echo "Setup Complete"
echo "=============================================="
echo "End time: $(date)"
echo "Next step: submit the experiment job with:"
echo "  sbatch <path-to-repository>/experiments/csf3_cloud_gpu.slurm"
echo "=============================================="

conda deactivate
