#!/bin/bash

# Define variables for easy updating
DATASET="gossipcop"  # "gossipcop" or "politifact"
N_TRIALS=200
EPOCHS=125
PATIENCE=25
DEVICE="cuda:7"  # Change to your preferred GPU device
OUTPUT_FILE="optuna_output.log"

# Create timestamp for unique log filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="optuna_${DATASET}_${TIMESTAMP}.log"

echo "Starting hyperparameter tuning for $DATASET with $N_TRIALS trials"
echo "Output will be saved to $OUTPUT_FILE"

# Run the hyperparameter tuning script with nohup
nohup python ../hyperparameter_tune.py \
  --dataset $DATASET \
  --n_trials $N_TRIALS \
  --epochs $EPOCHS \
  --patience $PATIENCE \
  --device $DEVICE \
  > $OUTPUT_FILE 2>&1 &

# Get the process ID
PID=$!
echo "Process started with PID: $PID"
echo "Use 'tail -f $OUTPUT_FILE' to monitor progress"
echo "Use 'kill $PID' to stop the process if needed"