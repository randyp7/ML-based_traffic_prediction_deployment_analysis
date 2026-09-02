#!/bin/bash --login
#===============================================================================
# CSF3 GPU Benchmark Job Script
# Sets up environment (if needed) and runs the full benchmark suite
# Fog-matched resources: A100 capped at 16GB VRAM, 12 cores, 32GB RAM
#===============================================================================

### GPU Partition - A100 (80GB) with software memory limit to 16GB
#SBATCH -p gpuA              # A100 (80GB) GPUs [up to 12 CPU cores per GPU]

### Resources — fog-matched configuration
#SBATCH -G 1                 # 1 GPU (matches fog: 1x RTX 4090)
#SBATCH -n 12                # 12 CPU cores (max per A100; fog has 14)
#SBATCH -t 1-0               # 1 day wallclock limit
#SBATCH --mem=32G            # 32GB RAM (matches fog)

### Job identification
#SBATCH -J benchmark_csf     # Job name
#SBATCH -o benchmark_csf_%j.out  # Standard output log
#SBATCH -e benchmark_csf_%j.err  # Standard error log

#===============================================================================
# Environment Setup
#===============================================================================

module purge
module load libs/cuda

echo "=============================================="
echo "CSF3 Benchmark Job Started"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS with IDs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_NTASKS cores"
echo "Memory: 32GB (fog-matched)"
echo "GPU Memory Limit: 16GB (fog-matched)"
echo "=============================================="

# Create conda environment if it doesn't exist
if ! conda info --envs | grep -q "benchmark"; then
    echo ""
    echo "Creating conda environment 'benchmark'..."
    conda create -n benchmark python=3.12 -y
    conda activate benchmark
    echo "Installing TensorFlow with GPU support..."
    pip install --no-cache-dir "tensorflow[and-cuda]"
    echo "Installing benchmark dependencies..."
    pip install --no-cache-dir pandas pyarrow scikit-learn geopy psutil
    echo "Environment setup complete."
else
    echo "Conda environment 'benchmark' already exists, activating..."
    conda activate benchmark
fi

# Verify GPU availability
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'TensorFlow GPUs detected: {len(gpus)}'); [print(f'  - {g.name}') for g in gpus]"

#===============================================================================
# Run Benchmark Suite
#===============================================================================

BENCHMARK_DIR=<path-to-repository>
cd $BENCHMARK_DIR

echo ""
echo "=============================================="
echo "Starting Full Benchmark Suite"
echo "Platform: csf3_cloud_gpu"
echo "Sensor tiers: tiny (1), small (10), medium (27), full (37)"
echo "Runs per tier: 3"
echo "Total experiments: 12"
echo "GPU memory capped at 16GB (matching fog RTX 4090)"
echo "=============================================="
echo ""

python -m experiments.runner \
    --platform csf3_cloud_gpu \
    --tiers tiny small medium full \
    --runs 3 \
    --gpu-memory-mb 16384

#===============================================================================
# Completion
#===============================================================================

echo ""
echo "=============================================="
echo "Benchmark Completed"
echo "=============================================="
echo "End time: $(date)"
echo "Results saved to: $BENCHMARK_DIR/src/results/csf3_cloud_gpu/"
echo "=============================================="

# List generated result files
echo ""
echo "Generated files:"
ls -la src/results/csf3_cloud_gpu/

conda deactivate
