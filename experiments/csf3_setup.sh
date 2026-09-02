#!/bin/bash
# CSF3 one-time setup: create conda environment with TensorFlow GPU
# Run this ONCE on a GPU interactive node:
#   srun -p gpuA -G 1 -n 1 -t 0-1 --pty bash
#   bash <path-to-repository>/experiments/csf3_setup.sh

set -e

echo "=== CSF3 Experiment Setup ==="

# Load Anaconda
module purge

# Create conda environment
echo "Creating conda environment 'experiment'..."
conda create -n traffic_pred python=3.12 -y

# Activate
conda activate traffic_pred

# Install TensorFlow with CUDA support
echo "Installing TensorFlow with GPU support..."
pip install --no-cache-dir "tensorflow[and-cuda]"

# Install experiment dependencies
echo "Installing experiment dependencies..."
pip install --no-cache-dir pandas pyarrow scikit-learn geopy psutil

# Verify GPU detection
echo ""
echo "=== Verifying GPU detection ==="
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs found: {len(gpus)}')
for g in gpus:
    print(f'  {g}')
if not gpus:
    print('WARNING: No GPU detected!')
else:
    print('GPU setup OK')
"

echo ""
echo "=== Setup complete ==="
echo "You can now submit the experiment job with:"
echo "  sbatch <path-to-repository>/experiments/csf3_cloud_gpu.slurm"
