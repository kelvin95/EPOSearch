#!/bin/bash

#SBATCH --job-name=mtl_nntd                  # Job name
#SBATCH --mail-type=ALL                      # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=amanjitsk@cs.toronto.edu # Where to send mail
#SBATCH --ntasks=1                           # Run a single task
#SBATCH --mem=8GB                            # Job Memory
#SBATCH --cpus-per-task=8                    # Number of CPUs
#SBATCH --partition=t4v1,p100,t4v2,rtx6000   # Which partition
#SBATCH --gres=gpu:1                         # Number of GPUs.
#SBATCH --output=vlogs/job_%A-%a.log         # Standard output and error log
#SBATCH --array=0-13                         # Array range

echo Running on $(hostname)

(while true; do
  nvidia-smi
  sleep 120
done) &

DSET="${1:-celeba}"
ARCH="${2:-lenet}"
# number of random seeds
NUM_RUNS="${3:-1}"

# array of methods
A=(epo
  graddrop
  graddrop_random
  graddrop_deterministic
  gradnorm
  gradvacc
  gradortho
  gradalign
  individual
  itmtl
  linscalar
  mgda
  pcgrad
  pmtl)

cd /h/amanjitsk/projects/EPOSearch/multiMNIST
for _ in $(seq $NUM_RUNS); do
  poetry run python train.py \
    --dset="$DSET" --arch="$ARCH" \
    --seed="$RANDOM" --solver="${A[$SLURM_ARRAY_TASK_ID]}" \
    --outdir=/checkpoint/amanjitsk/runs
done
