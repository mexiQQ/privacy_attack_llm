#!/bin/bash

# Nameme of the script to submit
SCRIPT_NAME="sbatch_GPU_31.sh"

while true; do
    # Submit the script using sbatch and capture the job ID
    JOB_OUTPUT=$(sbatch $SCRIPT_NAME)
    JOB_ID=$(echo $JOB_OUTPUT | awk '{print $NF}')
    
    # Sleep for a short duration to give the job a chance to appear in the queue
    sleep 5
    
    # Check if the job is in squeue
    if squeue -j $JOB_ID | grep -q $JOB_ID; then
        echo "Job $JOB_ID successfully submitted and found in the queue!"
        break
    else
        echo "Job $JOB_ID not found in the queue. Retrying..."
    fi
done

