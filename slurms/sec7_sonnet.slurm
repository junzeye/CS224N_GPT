#!/bin/bash

#SBATCH -J dfp
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -t 0-5:30:00
#SBATCH -G 1
#SBATCH --nodelist=yen-gpu3
#SBATCH -o out/sonnets/%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=junze@stanford.edu

python3 sonnet_generation.py --use_gpu --epochs 1 --model_size gpt2-large