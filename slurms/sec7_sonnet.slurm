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

# Define an array of learning rates
LEARNING_RATES=(0.0001 0.00001 0.000001)

# Loop through GPUs and learning rates
for i in {0..2}; do
    LR=${LEARNING_RATES[$i]}
    echo "Running with learning rate: $LR"
    OUTPUT_PATH="predictions/generated_sonnets_lr_${LR}.txt"
    
    # Assign 3 non-overlapping cores to each process
    START_CORE=$((i*3))
    END_CORE=$((START_CORE+2))
    
    CUDA_VISIBLE_DEVICES=$i taskset -c $START_CORE-$END_CORE python3 sonnet_generation.py \
        --use_gpu \
        --epochs 1 \
        --model_size gpt2-large \
        --lr $LR \
        --sonnet_out $OUTPUT_PATH \
        > out/sonnets/batch_16_lr_${LR}.log 2>&1 &
done

wait

