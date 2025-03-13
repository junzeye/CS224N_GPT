declare -a LEARNING_RATES=("2e-05" "1e-05")
declare -a A1s=("1" "0.1" "5")
declare -a A2s=("1" "0.1" "5")
BATCH_SIZE=32
EPOCHS=1

# Create a task array with all combinations
TASKS=()
for lr in "${LEARNING_RATES[@]}"; do
    for a1 in "${A1s[@]}"; do
        for a2 in "${A2s[@]}"; do
            TASKS+=("$lr $a1 $a2 $BATCH_SIZE $EPOCHS")
        done
    done
done

NUM_GPUS=8
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
    
    # Create output directories if they don't exist
    mkdir -p ckpts/para out/paraphrase
    
    # Add debug info about what command we're running
    echo "$(date): DEBUG: Running command with CUDA_VISIBLE_DEVICES=${GPUS[$worker_id]}"
    
    # First run a quick test to see if the Python script exists and is executable
    if [ ! -f "paraphrase_detection_new.py" ]; then
        echo "$(date): ERROR: Python script paraphrase_detection_new.py not found!"
        return 1
    fi
    
    # Run the task with explicit error logging
    CUDA_VISIBLE_DEVICES=${GPUS[$worker_id]} \
    python3 paraphrase_detection_new.py \
        --use_gpu \
        --ckpt_path "ckpts/para/epoch_${epochs}-bs_${batch_size}-lr_${lr}-a1_${a1}-a2_${a2}.pt" \
        --batch_size "${batch_size}" \
        --epochs "${epochs}" \
        > "out/paraphrase/ft_lr_${lr}_a1_${a1}_a2_${a2}_batch_size_${batch_size}_epochs_${epochs}.log" 2>&1 &
    
    # Store the process ID
    PIDS[$worker_id]=$!
    
    echo "$(date): DEBUG: Process started with PID: ${PIDS[$worker_id]}"
    
    # Give the process a moment to start up properly
    sleep 2
    
    # Verify the process started correctly
    if ! kill -0 ${PIDS[$worker_id]} 2>/dev/null; then
        echo "$(date): ERROR: Task failed to start properly on worker $worker_id (GPU ${GPUS[$worker_id]})"
        echo "$(date): ERROR: Check log file: out/paraphrase/ft_lr_${lr}_a1_${a1}_a2_${a2}_batch_size_${batch_size}_epochs_${epochs}.log"
        # Capture the last few lines of the log to see the error
        if [ -f "out/paraphrase/ft_lr_${lr}_a1_${a1}_a2_${a2}_batch_size_${batch_size}_epochs_${epochs}.log" ]; then
            echo "$(date): ERROR LOG:"
            tail -n 20 "out/paraphrase/ft_lr_${lr}_a1_${a1}_a2_${a2}_batch_size_${batch_size}_epochs_${epochs}.log"
        fi
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

# Print the full command for the first task for debugging
echo "$(date): DEBUG: First task command would be:"
if [ $total_tasks -gt 0 ]; then
    task_params=(${TASKS[0]})
    echo "CUDA_VISIBLE_DEVICES=0 python3 paraphrase_detection_new.py --use_gpu --filepath ckpts/para/epoch_${task_params[4]}-bs_${task_params[3]}-lr_${task_params[0]}-a1_${task_params[1]}-a2_${task_params[2]}.pt --batch_size ${task_params[3]} --epochs ${task_params[4]}"
fi

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
        # echo "$(date): DEBUG: No available workers, waiting..."
        sleep 10  # Check for available workers every 10 seconds
        continue
    fi
    
    echo "$(date): DEBUG: Found available worker $worker_id for task $task_index"
    
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