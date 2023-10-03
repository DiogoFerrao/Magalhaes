import json
from copy import deepcopy


# Define a function to create an augmentation entry with parameters
def create_augmentation_entry(name, params):
    return {"name": name, "params": params}


# Define the transformations to test
spec_transforms = ["Roll", "TimeMasking", "FreqMasking"]
waveform_transforms = [
    "AirAbsorption",
    "ClippingDistortion",
    "Gain",
    "AddGaussianNoise",
    "PitchShift",
    "LowPassFilter",
    "Oversampler",
    "BackgroundNoise",
    "TimeStretch",
]

# Define the base configuration with no augmentations
base_config = {
    "waveform_augmentations": [],
    "spectrogram_augmentations": [],
}

# Define a dictionary of example parameters for each transformation
transform_parameters = {
    "PitchShift": [
        {"p": 0.2, "min_semitones": -4.0, "max_semitones": 4.0},
        {"p": 0.2, "min_semitones": -3.0, "max_semitones": 3.0},
        {"p": 0.2, "min_semitones": -2.0, "max_semitones": 2.0},
        {"p": 0.2, "min_semitones": -1.0, "max_semitones": 1.0},
        {"p": 0.2, "min_semitones": -0.5, "max_semitones": 0.5},
    ],
    "AddGaussianNoise": [
        {"p": 0.2, "min_amplitude": 0.001, "max_amplitude": 0.015},
        {"p": 0.2, "min_amplitude": 0.002, "max_amplitude": 0.020},
        {"p": 0.2, "min_amplitude": 0.003, "max_amplitude": 0.025},
        {"p": 0.2, "min_amplitude": 0.004, "max_amplitude": 0.030},
        {"p": 0.2, "min_amplitude": 0.005, "max_amplitude": 0.035},
        {"p": 0.2, "min_amplitude": 0.006, "max_amplitude": 0.040},
    ],
    "AirAbsorption": [
        {"p": 0.2, "min_distance": 10.0, "max_distance": 50.0},
        {"p": 0.2, "min_distance": 15.0, "max_distance": 55.0},
        {"p": 0.2, "min_distance": 20.0, "max_distance": 60.0},
        {"p": 0.2, "min_distance": 25.0, "max_distance": 65.0},
        {"p": 0.2, "min_distance": 30.0, "max_distance": 70.0},
        {"p": 0.2, "min_distance": 35.0, "max_distance": 75.0},
    ],
    "ClippingDistortion": [
        {"p": 0.2, "min_percentile_threshold": 0, "max_percentile_threshold": 40},
        {"p": 0.2, "min_percentile_threshold": 10.0, "max_percentile_threshold": 50.0},
        {"p": 0.2, "min_percentile_threshold": 20.0, "max_percentile_threshold": 60.0},
        {"p": 0.2, "min_percentile_threshold": 0.0, "max_percentile_threshold": 30.0},
        {"p": 0.2, "min_percentile_threshold": 0.0, "max_percentile_threshold": 20.0},
        {"p": 0.2, "min_percentile_threshold": 0.0, "max_percentile_threshold": 10.0},
    ],
    "Gain": [
        {"p": 0.2, "min_gain_db": -12.0, "max_gain_db": 12.0},
        {"p": 0.2, "min_gain_db": -6.0, "max_gain_db": 6.0},
        {"p": 0.2, "min_gain_db": -3.0, "max_gain_db": 3.0},
        {"p": 0.2, "min_gain_db": -1.5, "max_gain_db": 1.5},
        {"p": 0.2, "min_gain_db": -0.5, "max_gain_db": 0.5},
    ],
    "LowPassFilter": [
        {"p": 0.2, "min_cutoff_freq": 150.0, "max_cutoff_freq": 7500.0},
        {"p": 0.2, "min_cutoff_freq": 600.0, "max_cutoff_freq": 10000.0},
        {"p": 0.2, "min_cutoff_freq": 0.0, "max_cutoff_freq": 5000.0},
        {"p": 0.2, "min_cutoff_freq": 3000.0 , "max_cutoff_freq": 12000.0},
        {"p": 0.2, "min_cutoff_freq": 1500.0, "max_cutoff_freq": 9000.0},
    ],
    "Roll": [
        {"shift_dims": (0, 0, 1), "dims": (0, 1, 2), "min": 0, "max": 250}
    ],
    "TimeMasking": [
        {"time_mask_param": 80},
        {"time_mask_param": 100},
        {"time_mask_param": 120},
        {"time_mask_param": 140},
        {"time_mask_param": 60},
        {"time_mask_param": 40},
        {"time_mask_param": 20}
    ],
    "FreqMasking": [
        {"freq_mask_param": 80},
        {"freq_mask_param": 100},
        {"freq_mask_param": 120},
        {"freq_mask_param": 140},
        {"freq_mask_param": 60},
        {"freq_mask_param": 40},
        {"freq_mask_param": 20}
    ],
    "Oversampler": [
        {"p": 0.2}
    ],
    "BackgroundNoise": [
        {"p": 0.2, "min_snr_db": 3.0, "max_snr_db": 30.0},
        {"p": 0.2, "min_snr_db": 6.0, "max_snr_db": 25.0},
        {"p": 0.2, "min_snr_db": 9.0, "max_snr_db": 20.0},
        {"p": 0.2, "min_snr_db": 12.0, "max_snr_db": 15.0},
    ],
    "TimeStretch": [
        {"p": 0.2, "min_rate": 0.8, "max_rate": 1.25},
        {"p": 0.2, "min_rate": 0.85, "max_rate": 1.15},
        {"p": 0.2, "min_rate": 0.9, "max_rate": 1.1},
        {"p": 0.2, "min_rate": 0.95, "max_rate": 1.05},
    ],
    # Add more transformations and their parameter variations as needed
}

# Create separate JSON configuration files for each transformation and parameter variation
for transform_name in spec_transforms + waveform_transforms:
    if transform_name in transform_parameters:
        # Get the list of parameter variations for the current transformation
        param_variations = transform_parameters[transform_name]
        print(f"Generating configuration files for {transform_name}...")
        print(param_variations)

        # Create a separate configuration file for each parameter variation
        for i, params in enumerate(param_variations):
            # Copy the base configuration to start fresh
            augmentation_config = deepcopy(dict(base_config))

            # Append the transformation entry with the current parameters
            if transform_name in spec_transforms:
                augmentation_config["spectrogram_augmentations"].append(
                    create_augmentation_entry(transform_name, params)
                )
            elif transform_name in waveform_transforms:
                augmentation_config["waveform_augmentations"].append(
                    create_augmentation_entry(transform_name, params)
                )

            # Write the configuration to a JSON file with a name based on the transformation and parameter index
            filename = f"augmentation_config_{transform_name}_{i}.json"
            with open(filename, "w") as f:
                json.dump(augmentation_config, f, indent=4)

            print(f"Configuration file '{filename}' generated successfully.")

# Generate a base case configuration file with no augmentations called Base_0
print("Generating configuration file for base case...")
filename = "augmentation_config_Base_0.json"

# Write the base configuration to a JSON file
json.dump(base_config, open(filename, "w"), indent=4)
