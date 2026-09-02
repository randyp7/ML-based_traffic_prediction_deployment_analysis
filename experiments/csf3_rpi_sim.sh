#!/bin/bash --login
#===============================================================================
# CSF3 Simulated Raspberry Pi Experiment (CPU-only)
# Simulates RPi 4B tier: 4 cores, 4GB RAM, no GPU
# NOTE: x86 architecture with much faster per-core performance than real RPi
#===============================================================================

### CPU-only Partition
#SBATCH -p serial            # CPU-only serial partition

### Resources — Raspberry Pi 4B matched
#SBATCH -n 4                 # 4 CPU cores (RPi 4B has 4 ARM Cortex-A72)
#SBATCH -t 2-0               # 2 day wallclock limit (CPU-only is slower)
#SBATCH --mem=4G             # 4GB RAM (RPi 4B 4GB model)

### Job identification
#SBATCH -J bench_rpi         # Job name
#SBATCH -o bench_rpi_%j.out  # Standard output log
#SBATCH -e bench_rpi_%j.err  # Standard error log

#===============================================================================
# Environment Setup
#===============================================================================

module purge

conda activate traffic_pred

echo "=============================================="
echo "CSF3 Simulated Raspberry Pi Experiment (CPU-only)"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_NTASKS cores (simulating RPi 4B quad-core)"
echo "Memory: 4GB (simulating RPi 4B)"
echo "GPU: None (CPU-only)"
echo "NOTE: x86 simulation — real RPi uses ARM Cortex-A72"
echo "=============================================="

#===============================================================================
# Run Experiment Suite
#===============================================================================

REPO_DIR=<path-to-repository>
cd $REPO_DIR

echo ""
echo "=============================================="
echo "Starting Experiment Suite"
echo "Platform: csf3_rpi_sim"
echo "Sensor tiers: tiny (1), small (10)"
echo "Runs per tier: 3"
echo "NOTE: Medium/Full likely OOM with 4GB RAM"
echo "=============================================="
echo ""

python -m experiments.runner \
    --platform csf3_rpi_sim \
    --tiers tiny small \
    --runs 3

#===============================================================================
# Completion
#===============================================================================

echo ""
echo "=============================================="
echo "Experiment Completed"
echo "=============================================="
echo "End time: $(date)"
echo "Results saved to: $REPO_DIR/src/results/csf3_rpi_sim/"
echo "=============================================="

echo ""
echo "Generated files:"
ls -la src/results/csf3_rpi_sim/

conda deactivate
