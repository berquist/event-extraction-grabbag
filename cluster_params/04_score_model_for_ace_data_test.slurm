#!/usr/bin/env bash

#SBATCH --job-name=04_score_model_for_ace_data_test
#SBATCH --output=04_score_model_for_ace_data_test.slurmout-%j
#SBATCH --error=04_score_model_for_ace_data_test.slurmerr-%j
#SBATCH --partition=gaia
#SBATCH --qos=gaia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1

# can't use -u because of conda
set -eo pipefail

if [[ -n "${SLURM_CPUS_PER_TASK}" ]]; then
    omp_threads="${SLURM_CPUS_PER_TASK}"
else
    omp_threads=1
fi
export OMP_NUM_THREADS="${omp_threads}"

source /nas/gaia/users/berquist/repositories/spack/share/spack/setup-env.sh
# When using `tensorflow-gpu`, paths to CUDA and CUDNN libraries are required
# by symbol lookup at runtime even if a GPU isn't going to be used.
spack load cuda@9.0.176
spack load cudnn@7.6.5.32-9.0-linux-x64
export CUDA_CACHE_DISABLE=1
source /nas/home/berquist/opt/apps/python/miniconda3/etc/profile.d/conda.sh
conda activate event-extraction-3.6
conda list
env | sort

python -m gaia_event_extraction.drivers.score_model_for_ace_data \
       "${SLURM_SUBMIT_DIR}"/04_score_model_for_ace_data_test.yaml
