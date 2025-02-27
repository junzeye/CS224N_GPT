#!/bin/bash

#SBATCH -J dfp
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -t 0-5:30:00
#SBATCH -G 1
#SBATCH --nodelist=yen-gpu3
#SBATCH -o out/%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=junze@stanford.edu

python3 paraphrase_detection.py --use_gpu --epochs 8
# python3 paraphrase_detection.py --use_gpu --epochs 5 --batch_size 32