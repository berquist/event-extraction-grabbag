#!/usr/bin/env bash

#SBATCH --job-name=01_run_bert_annotator_on_ace_train
#SBATCH --output=01_run_bert_annotator_on_ace_train.slurmout-%j
#SBATCH --error=01_run_bert_annotator_on_ace_train.slurmerr-%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:00:00

if [[ -n "${SLURM_CPUS_PER_TASK}" ]]; then
    omp_threads="${SLURM_CPUS_PER_TASK}"
else
    omp_threads=1
fi
export OMP_NUM_THREADS="${omp_threads}"

spack_root=/nas/gaia/users/berquist/repositories/spack
export SPACK_ROOT="${spack_root}"
source "${SPACK_ROOT}"/share/spack/setup-env.sh
# When using `tensorflow-gpu`, paths to CUDA and CUDNN libraries are required
# by symbol lookup at runtime even if a GPU isn't going to be used.
spack load cuda@9.0.176
spack load cudnn@7.6.5.32-9.0-linux-x64
export CUDA_CACHE_DISABLE=1
source /nas/home/berquist/opt/apps/python/miniconda3/etc/profile.d/conda.sh
conda activate event-extraction-3.6
conda list
env | sort

splitname=train
python -m gaia_event_extraction.drivers.run_bert_annotator_on_ace \
       "${SLURM_SUBMIT_DIR}"/01_run_bert_annotator_on_ace_${splitname}.params
