#!/bin/bash --login
#===============================================================================
# CSF3 Simulated Jetson Xavier NX Benchmark
# Simulates edge-intelligent tier: 6 cores, 8GB RAM, 1 GPU (8GB VRAM)
# NOTE: x86 architecture — actual Jetson uses ARM Carmel cores
#===============================================================================

### GPU Partition - A100 with memory capped to simulate Xavier NX
#SBATCH -p gpuA              # A100 (80GB) GPUs [up to 12 CPU cores per GPU]

### Resources — Jetson Xavier NX matched
#SBATCH -G 1                 # 1 GPU (Xavier NX has 384-core Volta)
#SBATCH -n 6                 # 6 CPU cores (Xavier NX has 6-core ARM Carmel)
#SBATCH -t 1-0               # 1 day wallclock limit
#SBATCH --mem=8G             # 8GB RAM (Xavier NX has 8GB unified memory)

### Job identification
#SBATCH -J bench_jetson      # Job name
#SBATCH -o bench_jetson_%j.out  # Standard output log
#SBATCH -e bench_jetson_%j.err  # Standard error log

#===============================================================================
# Environment Setup
#===============================================================================

module purge
module load libs/cuda

conda activate benchmark

echo "=============================================="
echo "CSF3 Simulated Jetson Xavier NX Benchmark"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS with IDs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_NTASKS cores (simulating 6-core ARM Carmel)"
echo "Memory: 8GB (simulating Xavier NX unified memory)"
echo "GPU Memory Limit: 8GB (simulating Xavier NX GPU allocation)"
echo "NOTE: x86 simulation — real Jetson uses ARM architecture"
echo "=============================================="

# Verify GPU availability
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'TensorFlow GPUs detected: {len(gpus)}'); [print(f'  - {g.name}') for g in gpus]"

#===============================================================================
# Run Benchmark Suite
#===============================================================================

BENCHMARK_DIR=<path-to-repository>
cd $BENCHMARK_DIR

echo ""
echo "=============================================="
echo "Starting Benchmark Suite"
echo "Platform: csf3_jetson_sim"
echo "Sensor tiers: tiny (1), small (10), medium (27), full (37)"
echo "Runs per tier: 3"
echo "GPU memory capped at 8GB (simulating Xavier NX)"
echo "=============================================="
echo ""

python -m experiments.runner \
    --platform csf3_jetson_sim \
    --tiers tiny small medium full \
    --runs 3 \
    --gpu-memory-mb 8192

#===============================================================================
# Completion
#===============================================================================

echo ""
echo "=============================================="
echo "Benchmark Completed"
echo "=============================================="
echo "End time: $(date)"
echo "Results saved to: $BENCHMARK_DIR/src/results/csf3_jetson_sim/"
echo "=============================================="

echo ""
echo "Generated files:"
ls -la src/results/csf3_jetson_sim/

conda deactivate
