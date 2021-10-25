#!/usr/bin/env bash

#module load generic anaconda3/2020.11
module load volta nvidia/cuda11.2-cudnn8.1.0

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate exp

# @AdaMo basts 0shot
#sbatch t2t_adamo.job 0shot4basts java evaluate @  # 4871916
# c_bleu = 0.0 | s_bleu = 1.8 | meteor = 0.1 | rouge = 0.24
#sbatch t2t_adamo.job 0shot4basts python evaluate @  # 4871917
# c_bleu = 0.0 | s_bleu = 1.78 | meteor = 0.15 | rouge = 0.36
# @AdaMo basts basic
#sbatch t2t_adamo.job basic4basts java evaluate @  # 4871918
# c_bleu = 31.38 | s_bleu = 37.64 | meteor = 25.59 | rouge = 49.9
#sbatch t2t_adamo.job basic4basts python evaluate @  # 4871919
# c_bleu = 5.19 | s_bleu = 16.46 | meteor = 12.51 | rouge = 27.31

# @AdaMo sit 0shot
#sbatch t2t_adamo.job 0shot java evaluate @  # 4871920
# c_bleu = 0.0 | s_bleu = 1.79 | meteor = 0.12 | rouge = 0.22
#sbatch t2t_adamo.job 0shot python evaluate @  # 4871921
# c_bleu = 0.0 | s_bleu = 1.89 | meteor = 0.07 | rouge = 0.13
# @AdaMo sit basic
#sbatch t2t_adamo.job basic java evaluate @  # 4871922
# c_bleu = 40.49 | s_bleu = 45.3 | meteor = 28.19 | rouge = 53.99
#sbatch t2t_adamo.job basic python evaluate @  # 4871923
# c_bleu = 26.52 | s_bleu = 33.85 | meteor = 21.68 | rouge = 41.25

# @AdaMo sit cp
#sbatch t2t_power.job cp-da-mlm java evaluate + 1  # 4872008
# c_bleu = 40.5 | s_bleu = 45.05 | meteor = 27.98 | rouge = 53.4
#sbatch t2t_power.job cp-da-clm java evaluate + 1  # 4871972
# c_bleu = 40.18 | s_bleu = 45.22 | meteor = 28.2 | rouge = 53.94
#sbatch t2t_power.job cp-da-both java evaluate + 1  # 4872009
# c_bleu = 40.62 | s_bleu = 45.05 | meteor = 28.16 | rouge = 53.82
#sbatch t2t_power.job cp-ta-mlm java evaluate + 1  # 4872010
# c_bleu = 40.61 | s_bleu = 45.27 | meteor = 28.08 | rouge = 53.72
#sbatch t2t_power.job cp-ta-clm java evaluate + 1  # 4872011
# c_bleu = 40.01 | s_bleu = 44.97 | meteor = 28.19 | rouge = 53.96
#sbatch t2t_power.job cp-ta-both java evaluate + 1  # 4872012
# c_bleu = 40.45 | s_bleu = 44.93 | meteor = 27.98 | rouge = 53.64
# @AdaMo sit cp
#sbatch t2t_power.job cp-da-mlm python evaluate + 1  # 4872013
# c_bleu = 26.47 | s_bleu = 33.83 | meteor = 21.55 | rouge = 40.99
#sbatch t2t_power.job cp-da-clm python evaluate + 1  # 4872014
# c_bleu = 26.35 | s_bleu = 33.68 | meteor = 21.73 | rouge = 41.41
#sbatch t2t_power.job cp-da-both python evaluate + 1  # 4872015
# c_bleu = 26.35 | s_bleu = 33.61 | meteor = 21.46 | rouge = 40.98
#sbatch t2t_power.job cp-ta-mlm python evaluate + 1  # 4872016
# c_bleu = 27.1 | s_bleu = 34.43 | meteor = 22.25 | rouge = 42.53
#sbatch t2t_power.job cp-ta-clm python evaluate + 1  # 4872017
# c_bleu = 26.53 | s_bleu = 33.8 | meteor = 21.89 | rouge = 41.66
#sbatch t2t_power.job cp-ta-both python evaluate + 1  # 4872018
# c_bleu = 26.86 | s_bleu = 34.21 | meteor = 22.16 | rouge = 42.37
# @AdaMo sit cp
#sbatch t2t_power.job cp-da-mlm java evaluate + 2  # 4872019
# c_bleu = 40.61 | s_bleu = 45.18 | meteor = 28.03 | rouge = 53.49
#sbatch t2t_power.job cp-da-clm java evaluate + 2  # 4872020
# c_bleu = 39.82 | s_bleu = 44.97 | meteor = 28.1 | rouge = 53.95
#sbatch t2t_power.job cp-da-both java evaluate + 2  # 4872021
# c_bleu = 40.39 | s_bleu = 44.94 | meteor = 27.97 | rouge = 53.4
#sbatch t2t_power.job cp-ta-mlm java evaluate + 2  # 4872022
# c_bleu = 40.55 | s_bleu = 45.37 | meteor = 28.15 | rouge = 53.97
#sbatch t2t_power.job cp-ta-clm java evaluate + 2  # 4872042
# c_bleu = 40.49 | s_bleu = 45.46 | meteor = 28.36 | rouge = 54.27
#sbatch t2t_power.job cp-ta-both java evaluate + 2  # 4872024
# c_bleu = 40.51 | s_bleu = 45.09 | meteor = 28.07 | rouge = 53.68
# @AdaMo sit cp
#sbatch t2t_power.job cp-da-mlm python evaluate + 2  # 4872025
# c_bleu = 26.28 | s_bleu = 33.56 | meteor = 21.46 | rouge = 40.85
#sbatch t2t_power.job cp-da-clm python evaluate + 2  # 4872026
# c_bleu = 26.34 | s_bleu = 33.67 | meteor = 21.81 | rouge = 41.69
#sbatch t2t_power.job cp-da-both python evaluate + 2  # 4872027
# c_bleu = 26.5 | s_bleu = 33.76 | meteor = 21.78 | rouge = 41.53
#sbatch t2t_power.job cp-ta-mlm python evaluate + 2  # 4872043
# c_bleu = 26.96 | s_bleu = 34.39 | meteor = 22.08 | rouge = 42.18
#sbatch t2t_power.job cp-ta-clm python evaluate + 2  # 4872044
# c_bleu = 26.61 | s_bleu = 33.86 | meteor = 21.84 | rouge = 41.58
#sbatch t2t_power.job cp-ta-both python evaluate + 2  # 4872068
# c_bleu = 27.03 | s_bleu = 34.3 | meteor = 22.25 | rouge = 42.49

# @AdaMo sit if
#sbatch t2t_adamo.job if-da-ca java evaluate +  # 4871948
# c_bleu = 29.06 | s_bleu = 35.75 | meteor = 20.53 | rouge = 44.2
#sbatch t2t_adamo.job if-da-ce java evaluate +  # 4871949
# c_bleu = 41.21 | s_bleu = 46.33 | meteor = 29.11 | rouge = 55.59
#sbatch t2t_adamo.job if-da-ci java evaluate +  # 4871950
# c_bleu = 40.65 | s_bleu = 45.99 | meteor = 28.83 | rouge = 55.31
#sbatch t2t_adamo.job if-ta-ca java evaluate +  # 4871951
# c_bleu = 39.79 | s_bleu = 44.63 | meteor = 27.73 | rouge = 53.19
#sbatch t2t_adamo.job if-ta-ce java evaluate +  # 4871952
# c_bleu = 40.5 | s_bleu = 45.42 | meteor = 28.02 | rouge = 53.45
#sbatch t2t_adamo.job if-ta-ci java evaluate +  # 4871953
# c_bleu = 40.15 | s_bleu = 45.33 | meteor = 28.3 | rouge = 54.24
# @AdaMo sit if
#sbatch t2t_adamo.job if-da-ca python evaluate +  # 4871954
# c_bleu = 25.6 | s_bleu = 33.05 | meteor = 21.02 | rouge = 40.11
#sbatch t2t_adamo.job if-da-ce python evaluate +  # 4871955
# c_bleu = 28.37 | s_bleu = 35.44 | meteor = 23.25 | rouge = 44.28
#sbatch t2t_adamo.job if-da-ci python evaluate +  # 4871956
# c_bleu = 27.66 | s_bleu = 34.88 | meteor = 22.96 | rouge = 43.82
#sbatch t2t_adamo.job if-ta-ca python evaluate +  # 4871957
# c_bleu = 25.84 | s_bleu = 33.42 | meteor = 21.42 | rouge = 40.97
#sbatch t2t_adamo.job if-ta-ce python evaluate +  # 4872069
# c_bleu = 26.53 | s_bleu = 33.91 | meteor = 21.15 | rouge = 40.51
#sbatch t2t_adamo.job if-ta-ci python evaluate +  # 4871959
# c_bleu = 26.22 | s_bleu = 33.65 | meteor = 21.79 | rouge = 41.44

# @AdaMo sit noisy
#sbatch t2t_noisy.job basic java evaluate = 0.01  # 4872070
# c_bleu = 39.04 | s_bleu = 43.41 | meteor = 26.86 | rouge = 51.67
#sbatch t2t_noisy.job basic java evaluate = 0.04  # 4872071
# c_bleu = 40.62 | s_bleu = 45.33 | meteor = 28.18 | rouge = 53.86
#sbatch t2t_noisy.job basic java evaluate = 0.09  # 4872072
# c_bleu = 40.52 | s_bleu = 45.35 | meteor = 28.25 | rouge = 54.06
#sbatch t2t_noisy.job basic java evaluate = 0.16  # 4872050
# c_bleu = 40.37 | s_bleu = 44.92 | meteor = 27.93 | rouge = 53.36
#sbatch t2t_noisy.job basic java evaluate = 0.25  # 4872073
# c_bleu = 40.56 | s_bleu = 45.0 | meteor = 28.04 | rouge = 53.49
# @AdaMo sit noisy
#sbatch t2t_noisy.job basic python evaluate = 0.01  # 4872074
# c_bleu = 25.21 | s_bleu = 32.55 | meteor = 20.79 | rouge = 39.45
#sbatch t2t_noisy.job basic python evaluate = 0.04  # 4872075
# c_bleu = 26.65 | s_bleu = 34.03 | meteor = 21.95 | rouge = 41.83
#sbatch t2t_noisy.job basic python evaluate = 0.09  # 4872076
# c_bleu = 26.8 | s_bleu = 34.05 | meteor = 21.92 | rouge = 41.67
#sbatch t2t_noisy.job basic python evaluate = 0.16  # 4872077
# c_bleu = 26.78 | s_bleu = 34.04 | meteor = 21.89 | rouge = 41.71
#sbatch t2t_noisy.job basic python evaluate = 0.25  # 4872078
# c_bleu = 26.74 | s_bleu = 34.05 | meteor = 21.91 | rouge = 41.64

conda deactivate
