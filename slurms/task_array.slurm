#!/bin/bash

#SBATCH -J paraphrase
#SBATCH -p gpu
#SBATCH -c 13
#SBATCH -N 1
#SBATCH -t 0-10:00:00
#SBATCH -G 4
#SBATCH --nodelist=yen-gpu3
#SBATCH -o out/paraphrase/%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=junze@stanford.edu

# Define all hyperparameter combinations to try
declare -a LEARNING_RATES=("1e-04" "1e-05" "1e-06")
declare -a A1s=("1" "0.1" "5")
declare -a A2s=("1" "0.1" "5")

# Create a task array with all combinations
TASKS=()
for lr in "${LEARNING_RATES[@]}"; do
    for a1 in "${A1s[@]}"; do
        for a2 in "${A2s[@]}"; do
            TASKS+=("$lr $a1 $a2")
        done
    done
done

# Initialize tracking arrays
declare -a PIDS=(-1 -1 -1 -1)         # Process IDs for each worker
declare -a GPUS=(0 1 2 3)             # Available GPUs
declare -a CPU_SETS=("1-3" "4-6" "7-9" "10-12")  # CPU sets for each worker - leave core 0 for main process
declare -a TASK_INFO=("" "" "" "")    # Store task information for each worker

# Function to find an available worker
function find_available_worker {
    for i in {0..3}; do
        # If PID exists but process is not running
        if [ ${PIDS[$i]} -ne -1 ] && ! kill -0 ${PIDS[$i]} 2>/dev/null; then
            # This worker just finished its task
            echo "$(date): Worker $i (GPU ${GPUS[$i]}) finished job with PID ${PIDS[$i]} - ${TASK_INFO[$i]}" >&2
            PIDS[$i]=-1  # Reset PID
            echo $i
            return 0
        # If PID is marked as not assigned
        elif [ ${PIDS[$i]} -eq -1 ]; then
            echo $i
            return 0
        fi
    done
    echo "-1"
}

# Function to run a task on a specific worker
function run_task {
    local worker_id=$1
    local lr=$2
    local a1=$3
    local a2=$4
    
    # Store task information for logging
    TASK_INFO[$worker_id]="LR=${lr}, a1=${a1}, a2=${a2}"
    
    echo "$(date): Starting task with learning rate: $lr, a1: $a1, a2: $a2 on worker $worker_id (GPU ${GPUS[$worker_id]}, CPUs ${CPU_SETS[$worker_id]})"
    
    # Run the task with taskset to pin to specific CPUs and set CUDA_VISIBLE_DEVICES for GPU
    taskset -c ${CPU_SETS[$worker_id]} \
    CUDA_VISIBLE_DEVICES=${GPUS[$worker_id]} \
    python3 paraphrase_detection_new.py \
        --use_gpu \
        --filepath ${lr}-${a1}-${a2}-paraphrase-ft.pt \
        > out/paraphrase/ft_lr_${lr}_a1_${a1}_a2_${a2}.log 2>&1 &
    
    # Store the process ID
    PIDS[$worker_id]=$!
    echo "$(date): Task started with PID: ${PIDS[$worker_id]}"
}

# Main execution loop
task_index=0
total_tasks=${#TASKS[@]}

echo "$(date): Starting execution with $total_tasks total tasks on 4 workers"
echo "Worker check interval: 10 seconds"

# Start initial batch of tasks (up to 4)
for i in {0..3}; do
    if [ $task_index -lt $total_tasks ]; then
        # Parse the task string to get parameters
        task_params=(${TASKS[$task_index]})
        lr=${task_params[0]}
        a1=${task_params[1]}
        a2=${task_params[2]}
        
        run_task $i "$lr" "$a1" "$a2"
        ((task_index++))
    fi
done

# Process remaining tasks as workers become available
while [ $task_index -lt $total_tasks ]; do
    # Find an available worker
    worker_id=$(find_available_worker)
    
    # If no worker is available, wait a bit and check again
    if [ $worker_id -eq -1 ]; then
        sleep 10  # Check for available workers every 10 seconds
        continue
    fi
    
    # Parse the task string to get parameters
    task_params=(${TASKS[$task_index]})
    lr=${task_params[0]}
    a1=${task_params[1]}
    a2=${task_params[2]}
    
    run_task $worker_id "$lr" "$a1" "$a2"
    ((task_index++))
done

echo "$(date): All tasks submitted, waiting for completion..."
# Check and report on remaining running tasks
while true; do
    still_running=false
    for i in {0..3}; do
        if [ ${PIDS[$i]} -ne -1 ] && kill -0 ${PIDS[$i]} 2>/dev/null; then
            still_running=true
        elif [ ${PIDS[$i]} -ne -1 ]; then
            echo "$(date): Worker $i (GPU ${GPUS[$i]}) finished job with PID ${PIDS[$i]} - ${TASK_INFO[$i]}"
            PIDS[$i]=-1
        fi
    done
    
    if [ "$still_running" = false ]; then
        break
    fi
    
    sleep 10
done

echo "$(date): All tasks completed successfully!"