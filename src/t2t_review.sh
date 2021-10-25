#!/usr/bin/env bash

module load generic anaconda3/2020.11
#module load volta nvidia/cuda11.2-cudnn8.1.0

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate exp

sbatch t2t_review.job

conda deactivate
