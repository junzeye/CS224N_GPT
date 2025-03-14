#!/bin/bash

#SBATCH -J dfp
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -t 0-02:00:00
#SBATCH -G 1
#SBATCH --nodelist=yen-gpu3
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=junze@stanford.edu

echo "Running sonnet generation with learning rate: $LR, batch size: $BATCH_SIZE, epochs: $EPOCHS..."

python3 sonnet_generation.py \
    --use_gpu \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --filepath ckpts/sonnets/${SLURM_JOB_NAME}.pt \
    --held_out_sonnet_path data/sonnets_held_out.txt \
    --sonnet_out predictions/sonnets/${SLURM_JOB_NAME}.txt

# use held out dev set for evaluation (so that we can calculate CHRF score locally)