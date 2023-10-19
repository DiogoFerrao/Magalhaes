#!/bin/bash

# Check if the directory argument is provided
if [[ -z "$1" ]]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Set the directory from the first command line argument
DIR="$1"

# Check if the provided directory exists
if [[ ! -d "$DIR" ]]; then
    echo "Error: Directory '$DIR' does not exist."
    exit 1
fi

# Loop through every file in the directory
for file in "$DIR"/*; do
    # Check if the current item is a file
    if [[ -f "$file" ]]; then
        # Check if the file name does NOT contain any of the specified strings
        if [[ ! "$file" =~ "config_BackgroundNoise_AirAbsorption_CI" && 
              ! "$file" =~ "config_BackgroundNoise_AirAbsorption_TimeStretch_CI" && 
              ! "$file" =~ "config_SpecAugment_AirAbsorption_ClippingDistortion_TimeStretch_CI" && 
              ! "$file" =~ "config_SpecAugment_BackgroundNoise_AirAbsorption_TimeStretch_CI" && 
              ! "$file" =~ "config_SpecAugment_BackgroundNoise_ClippingDistortion_TimeStretch_CI" && 
              ! "$file" =~ "config_TimeStretch_CI" ]]; then
            # Delete the file
            rm "$file"
            echo "Deleted: $file"
        fi
    fi
done

echo "Cleanup complete."
