#!/usr/bin/env bash

#SBATCH --job-name=01_run_bert_annotator_on_ace_train
#SBATCH --output=01_run_bert_annotator_on_ace_train.slurmout-%j
#SBATCH --error=01_run_bert_annotator_on_ace_train.slurmerr-%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-04:00:00

if [[ -n "${SLURM_CPUS_PER_TASK}" ]]; then
    omp_threads="${SLURM_CPUS_PER_TASK}"
else
    omp_threads=1
fi
export OMP_NUM_THREADS="${omp_threads}"

# When using `tensorflow-gpu`, paths to CUDA and CUDNN libraries are required
# by symbol lookup at runtime even if a GPU isn't going to be used.
export CUDA_HOME="$(spack location -i cuda@9.0.176)"
export CUDNN_HOME="$(spack location -i cudnn@7.3.0)"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDNN_HOME/lib64:$LD_LIBRARY_PATH"
source /nas/home/berquist/opt/apps/python/miniconda3/etc/profile.d/conda.sh
conda activate event-extraction-3.6
conda list
env | sort

splitname=train
python -m gaia_event_extraction.drivers.run_bert_annotator_on_ace \
       "${SLURM_SUBMIT_DIR}"/01_run_bert_annotator_on_ace_${splitname}.params
