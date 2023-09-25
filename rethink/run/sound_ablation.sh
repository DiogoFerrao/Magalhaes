#!/bin/bash

# Directory where your configuration files are located
config_dir="sound_ablation_configs"

# List of configuration files
config_files=($config_dir/*.json)

# Loop through each configuration file
for config_file in "${config_files[@]}"; do
    echo "Training and Testing with $config_file"

    # Training command
    python train.py --config_path "$config_file"

    # Sleep for 30 seconds before testing
    echo "Waiting for 30 seconds before testing..."
    sleep 30

    # Testing command
    python evaluate.py --config_path "$config_file"

    # Sleep for 10 minutes before the next run
    echo "Waiting for 10 minutes before the next run..."
    sleep 600
done
