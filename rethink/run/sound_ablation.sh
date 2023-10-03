#!/bin/bash

# Directory where your configuration files are located
config_dir="sound_ablation_configs"

# List of configuration files
config_files=($config_dir/*.json)

# Get the total number of configuration files
total_configs=${#config_files[@]}

# Loop through each configuration file
for ((i = 0; i < total_configs; i++)); do
    config_file="${config_files[$i]}"
    echo "Training and Testing with $config_file"

    # Training command
    python train.py --config_path "$config_file"

    # Sleep for 30 seconds before testing
    echo "Waiting for 30 seconds before testing..."
    sleep 30

    # Testing command
    python evaluate.py --config_path "$config_file"

    # Check if it's not the last iteration, then sleep for 5 minutes
    if [ $i -lt $((total_configs - 1)) ]; then
        echo "Waiting for 5 minutes before the next run..."
        sleep 300
    else
        echo "No need to wait for the last iteration."
    fi
done
