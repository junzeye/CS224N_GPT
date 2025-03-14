#!/bin/bash

#SBATCH -J vanilla
#SBATCH -p gpu
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -t 0-10:00:00
#SBATCH -G 4
#SBATCH --nodelist=yen-gpu2
#SBATCH -o out/%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=junze@stanford.edu

# IMPORTANT NOTE: Still need to clean up a couple of things for this template to work! Don't run this yet

# Define all configurable parameters at the top
declare -a LEARNING_RATES=("1e-04" "1e-05" "2e-05" "5e-05" "1e-06")
BATCH_SIZE=32
EPOCHS=2
NUM_GPUS=4

# Define the Python command template
PYTHON_CMD_TEMPLATE='python3 paraphrase_detection.py \
    --lr "${lr}" \
    --batch_size "${batch_size}" \
    --epochs "${epochs}" \
    --model_size "gpt2" \
    --use_gpu \
    --filepath "ckpts/para/baseline/epoch_${epochs}-bs_${batch_size}-lr_${lr}.pt" \
    > "out/paraphrase/ft_lr_${lr}_batch_size_${batch_size}_epochs_${epochs}.log" 2>&1'

# Create a task array with all combinations
TASKS=()
for lr in "${LEARNING_RATES[@]}"; do
    TASKS+=("$lr $BATCH_SIZE $EPOCHS")
done

# Initialize tracking arrays with dynamic size based on NUM_GPUS
declare -a PIDS=()
declare -a GPUS=()
declare -a RUNNING_TASKS_INFO=()
# Fill arrays with initial values
for ((i=0; i<$NUM_GPUS; i++)); do
    PIDS[$i]=-1         # Initialize all PIDs to -1 (not running)
    GPUS[$i]=$i         # Assign GPU IDs 0 through NUM_GPUS-1
    RUNNING_TASKS_INFO[$i]=""    # Initialize task info as empty
done


# Function to find an available worker
function find_available_worker {
    for ((i=0; i<$NUM_GPUS; i++)); do
        # If PID exists but process is not running
        if [ ${PIDS[$i]} -ne -1 ]; then
            if ! kill -0 ${PIDS[$i]} 2>/dev/null; then
                # This worker just finished its task
                echo "$(date): Worker $i (GPU ${GPUS[$i]}) finished job with PID ${PIDS[$i]} - ${RUNNING_TASKS_INFO[$i]}" >&2
                PIDS[$i]=-1  # Reset PID
                echo $i
                return 0
            fi
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
    local batch_size=$5
    local epochs=$6
    
    # Store task information for logging
    RUNNING_TASKS_INFO[$worker_id]="LR=${lr}, a1=${a1}, a2=${a2}, batch_size=${batch_size}, epochs=${epochs}"
    
    echo "$(date): Starting task with learning rate: $lr, a1: $a1, a2: $a2, batch_size: $batch_size, epochs: $epochs on worker $worker_id (GPU ${GPUS[$worker_id]})"
    echo "$(date): Running command with CUDA_VISIBLE_DEVICES=${GPUS[$worker_id]}"

    # Create output directories if they don't exist
    mkdir -p ckpts/para out/para
    
    # First run a quick test to see if the Python script exists and is executable
    if [ ! -f "paraphrase_detection_new.py" ]; then
        echo "$(date): ERROR: Python script paraphrase_detection_new.py not found!"
        return 1
    fi
    
    # Construct the command using the template and eval
    CMD="CUDA_VISIBLE_DEVICES=${GPUS[$worker_id]} ${PYTHON_CMD_TEMPLATE} &"
    echo "$(date): Running command: $(eval echo "$CMD")"    
    # Execute the command
    eval "$CMD"
    
    # Store the process ID
    PIDS[$worker_id]=$!
    
    echo "$(date): Process started with PID: ${PIDS[$worker_id]}"    
    sleep 2
    
    # Verify the process started correctly
    if ! kill -0 ${PIDS[$worker_id]} 2>/dev/null; then
        echo "$(date): ERROR: Task failed to start properly on worker $worker_id (GPU ${GPUS[$worker_id]})"
        PIDS[$worker_id]=-1
        return 1
    fi
    
    echo "$(date): Task started with PID: ${PIDS[$worker_id]}"
    return 0
}

# Main execution loop
task_index=0
total_tasks=${#TASKS[@]}

echo "$(date): Starting execution with $total_tasks total tasks on $NUM_GPUS workers"
echo "Worker check interval: 10 seconds"

# Start initial batch of tasks (up to NUM_GPUS)
for ((i=0; i<$NUM_GPUS; i++)); do
    if [ $task_index -lt $total_tasks ]; then
        # Parse the task string to get parameters
        task_params=(${TASKS[$task_index]})
        lr=${task_params[0]}
        a1=${task_params[1]}
        a2=${task_params[2]}
        batch_size=${task_params[3]}
        epochs=${task_params[4]}
        
        run_task $i "$lr" "$a1" "$a2" "$batch_size" "$epochs"
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
    
    echo "$(date): Found available worker $worker_id for task $task_index"
    
    # Parse the task string to get parameters
    task_params=(${TASKS[$task_index]})
    lr=${task_params[0]}
    a1=${task_params[1]}
    a2=${task_params[2]}
    batch_size=${task_params[3]}
    epochs=${task_params[4]}
    
    run_task $worker_id "$lr" "$a1" "$a2" "$batch_size" "$epochs"
    task_status=$?
    if [ $task_status -ne 0 ]; then
        echo "$(date): ERROR: Task $task_index failed to start properly. Skipping."
    fi
    ((task_index++))
done

echo "$(date): All tasks submitted, waiting for completion..."
# Check and report on remaining running tasks
while true; do
    still_running=false
    for ((i=0; i<$NUM_GPUS; i++)); do
        if [ ${PIDS[$i]} -ne -1 ] && kill -0 ${PIDS[$i]} 2>/dev/null; then
            still_running=true
        elif [ ${PIDS[$i]} -ne -1 ]; then
            echo "$(date): Worker $i (GPU ${GPUS[$i]}) finished job with PID ${PIDS[$i]} - ${RUNNING_TASKS_INFO[$i]}"
            PIDS[$i]=-1
        fi
    done
    
    if [ "$still_running" = false ]; then
        break
    fi
    
    sleep 10
done

echo "$(date): All tasks completed successfully!"