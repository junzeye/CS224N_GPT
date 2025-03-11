#!/bin/bash

#SBATCH -J dfp
#SBATCH -p gpu
#SBATCH -c 9
#SBATCH -N 1
#SBATCH -t 0-5:00:00
#SBATCH -G 3
#SBATCH --nodelist=yen-gpu3
#SBATCH -o out/sonnets/%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=junze@stanford.edu

LEARNING_RATES=("0.0001" "1e-05" "1e-06")

for i in {0..2}; do
    LR=${LEARNING_RATES[$i]}
    echo "Evaluating dev set with learning rate: $LR"
    CUDA_VISIBLE_DEVICES=$i python3 sonnet_generation.py \
        --use_gpu \
        --held_out_sonnet_path data/sonnets_held_out_dev.txt \
        --sonnet_out predictions/generated_sonnets_dev_lr_${LR}.txt \
        --filepath 0_1-${LR}-sonnet.pt \
        > out/sonnets/dev_eval_lr_${LR}.log 2>&1 &
done

wait