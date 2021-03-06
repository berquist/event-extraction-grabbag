#!/usr/bin/env bash

#SBATCH --job-name=02_train_tbnnam_lstm_last_ace_single_split
#SBATCH --output=02_train_tbnnam_lstm_last_ace_single_split.slurmout-%j
#SBATCH --error=02_train_tbnnam_lstm_last_ace_single_split.slurmerr-%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --time=0-24:00:00

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

python -m gaia_event_extraction.model_trainers.train_event_extraction_without_triggers_lstm_last \
       "${SLURM_SUBMIT_DIR}"/02_train_tbnnam_lstm_last_ace_single_split.yaml
