#!/bin/bash

# Path to the existing slurm script
SLURM_SCRIPT_PATH="slurms/sec7_sonnet.slurm"

# Define hyperparameters to vary
# declare -a LEARNING_RATES=("1e-4" "1e-5" "1e-6")
# declare -a BATCH_SIZES=("16" "32")
# declare -a EPOCHS=("2" "4")

# look at the performance of zero-shot, no finetune
declare -a LEARNING_RATES=("1e-4")
declare -a BATCH_SIZES=("16")
declare -a EPOCHS=("2")

# Counter for job tracking
job_count=0

echo "Starting job submissions..."

# Loop through all hyperparameter combinations
for lr in "${LEARNING_RATES[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        for ep in "${EPOCHS[@]}"; do
            JOB_NAME="exp_lr${lr}_bs${bs}_ep${ep}"            
            echo "Submitting job: $JOB_NAME with lr=$lr, batch_size=$bs, epochs=$ep"
            
            # Submit the slurm job with parameters
            sbatch --job-name="$JOB_NAME" \
                   --output="out/sonnets/$JOB_NAME.out" \
                   --export=ALL,LR="$lr",BATCH_SIZE="$bs",EPOCHS="$ep" \
                   "$SLURM_SCRIPT_PATH"

            ((job_count++))
            sleep 1
        done
    done
done

echo "Submitted $job_count jobs in total."