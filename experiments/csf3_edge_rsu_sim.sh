#!/bin/bash --login
#===============================================================================
# CSF3 Simulated Edge RSU Experiment (CPU-only)
# Simulates edge-RSU tier: 2 cores, 4GB RAM, no GPU
# Matches AWS t2.medium / roadside unit specs
# NOTE: x86 architecture with faster per-core performance than real RSU
#===============================================================================

### CPU-only Partition
#SBATCH -p serial            # CPU-only serial partition

### Resources — Edge RSU matched
#SBATCH -n 2                 # 2 CPU cores (t2.medium has 2 vCPUs)
#SBATCH -t 2-0               # 2 day wallclock limit (CPU-only is slower)
#SBATCH --mem=4G             # 4GB RAM (t2.medium has 4GB)

### Job identification
#SBATCH -J bench_rsu         # Job name
#SBATCH -o bench_rsu_%j.out  # Standard output log
#SBATCH -e bench_rsu_%j.err  # Standard error log

#===============================================================================
# Environment Setup
#===============================================================================

module purge

conda activate traffic_pred

echo "=============================================="
echo "CSF3 Simulated Edge RSU Experiment (CPU-only)"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_NTASKS cores (simulating 2-core RSU)"
echo "Memory: 4GB (simulating RSU/t2.medium)"
echo "GPU: None (CPU-only)"
echo "NOTE: x86 simulation — real RSU may use ARM"
echo "=============================================="

#===============================================================================
# Run Experiment Suite
#===============================================================================

REPO_DIR=<path-to-repository>
cd $REPO_DIR

echo ""
echo "=============================================="
echo "Starting Experiment Suite"
echo "Platform: csf3_edge_rsu_sim"
echo "Sensor tiers: tiny (1), small (10)"
echo "Runs per tier: 3"
echo "NOTE: Medium/Full likely OOM with 4GB RAM"
echo "=============================================="
echo ""

python -m experiments.runner \
    --platform csf3_edge_rsu_sim \
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
echo "Results saved to: $REPO_DIR/src/results/csf3_edge_rsu_sim/"
echo "=============================================="

echo ""
echo "Generated files:"
ls -la src/results/csf3_edge_rsu_sim/

conda deactivate
