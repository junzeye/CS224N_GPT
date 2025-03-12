#!/bin/bash

#SBATCH -J dfp
#SBATCH -p gpu
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -t 0-3:30:00
#SBATCH -G 2
#SBATCH --nodelist=yen-gpu1
#SBATCH -o out/%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=junze@stanford.edu

# declare learning rates as an array
declare -a LRs=("2e-5" "1e-4")

for i in {0..1}; do
    CUDA_VISIBLE_DEVICES=$i python3 paraphrase_detection.py \
        --model_size gpt2 --epochs 2 \
        --lr ${LRs[$i]} \
        --use_gpu \
        --filepath ckpts/para/baseline/epochs_2_lr_${LRs[$i]}.pt \
        > out/para/baseline/lr_${LRs[$i]}.log 2>&1 &
    echo "started job ${i}!"
done

# wait for all subprocesses to finish
wait
echo "All jobs finished!"