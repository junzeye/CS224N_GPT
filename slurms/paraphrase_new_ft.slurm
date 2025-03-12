#!/bin/bash

#SBATCH -J dfp
#SBATCH -p gpu
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -t 0-6:30:00
#SBATCH -G 4
#SBATCH --nodelist=yen-gpu1
#SBATCH -o out/%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=junze@stanford.edu

# # just for initial testing
# CUDA_VISIBLE_DEVICES=0 python3 paraphrase_detection_new.py \
#     --model_size gpt2 --epochs 1 \
#     --lr 1e-5 --a1 0.1 --a2 0.01 \
#     --use_gpu \
#     --ckpt_path ckpts/para/lr_1e-5_a1_0.1_a2_0.01.pt

declare -a A1s=("0.001" "0.01" "0.1" "0")
declare -a A2s=("0.001" "0.01" "0.1" "0")
declare -a LRs=("1e-5" "1e-5" "1e-5" "2e-5")
declare -a BATCH_SIZES=("32" "32" "32" "32")
EPOCHS=2

for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=$i python3 paraphrase_detection_new.py \
        --model_size gpt2 --epochs ${EPOCHS} --batch_size ${BATCH_SIZES[$i]} \
        --lr ${LRs[$i]} --a1 ${A1s[$i]} --a2 ${A2s[$i]} \
        --use_gpu \
        --ckpt_path ckpts/para/epochs_${EPOCHS}_lr_${LRs[$i]}_a1_${A1s[$i]}_a2_${A2s[$i]}_bs_${BATCH_SIZES[$i]}.pt \
        > out/para/epochs_${EPOCHS}_lr_${LRs[$i]}_a1_${A1s[$i]}_a2_${A2s[$i]}_bs_${BATCH_SIZES[$i]}.log 2>&1 &
    echo "started job ${i}!"
done

# wait for all subprocesses to finish
wait
echo "All jobs finished!"
