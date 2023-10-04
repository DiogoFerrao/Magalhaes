import json
from copy import deepcopy
import argparse
import itertools

# Define a function to create an augmentation entry with parameters
def create_augmentation_entry(name, params):
    return {"name": name, "params": params}

def combine_all(transform_parameters: dict) -> dict:
    spec_transforms = list(transform_parameters["spectrogram"].keys())
    waveform_transforms = list(transform_parameters["waveform"].keys())

    all_transforms = {**transform_parameters["spectrogram"], **transform_parameters["waveform"]}

    # Define the base configuration with no augmentations
    base_config = {
        "waveform_augmentations": [],
        "spectrogram_augmentations": [],
    }

    augmentation_configs = {}

    # Generate all combinations of augmentations
    for num_augmentations in range(1, len(all_transforms) + 1):
        augmentation_combinations = itertools.combinations(all_transforms.keys(), num_augmentations)

        for i, combination in enumerate(augmentation_combinations):
            augmentation_config = deepcopy(base_config)
            transform_name = "_".join(combination)  # Concatenate the names of augmentations

            for augmentation_name in combination:
                param_variations = all_transforms[augmentation_name]

                for params in param_variations:
                    if augmentation_name in spec_transforms:
                        augmentation_config["spectrogram_augmentations"].append(
                            create_augmentation_entry(augmentation_name, params)
                        )
                    elif augmentation_name in waveform_transforms:
                        augmentation_config["waveform_augmentations"].append(
                            create_augmentation_entry(augmentation_name, params)
                        )

            export_name = f"augmentation_config_{transform_name}"
            augmentation_configs[export_name] = augmentation_config

    return augmentation_configs



def combine_one_each(transform_parameters: dict) -> dict:
    spec_transforms = list(transform_parameters["spectrogram"].keys())
    waveform_transforms = list(transform_parameters["waveform"].keys())

    all_transforms = { **transform_parameters["spectrogram"], **transform_parameters["waveform"]}

    # Define the base configuration with no augmentations
    base_config = {
        "waveform_augmentations": [],
        "spectrogram_augmentations": [],
    }
    
    configs = {}

    # Create separate JSON configuration files for each transformation and parameter variation
    for transform_name in spec_transforms + waveform_transforms:
        if transform_name in all_transforms:
            # Get the list of parameter variations for the current transformation
            param_variations = all_transforms[transform_name]
            print(f"Generating configuration files for {transform_name}...")

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
            
                export_name = f"augmentation_config_{transform_name}_{i}"

                # Add the config to the dict config on key export_name
                configs[export_name] = augmentation_config

    return configs


if __name__ == "__main__":
    # Receive command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentations_file", type=str, default="./config/augmentations.json")
    parser.add_argument("--output_dir", type=str, default="./preprocessing_configs")
    parser.add_argument("--combination_mode", type=str, choices=["all", "one_each"], default="one_each")

    args = parser.parse_args()

    # Read transformers_parameters from augmentations file
    with open(args.augmentations_file) as f:
        transformers_parameters = json.load(f)
    
    # TODO: a bit of a hack
    # If the transformation_parameters have a "Roll" entry under "spectrogram", convert the arrays to tuples
    if "Roll" in transformers_parameters["spectrogram"]:
        # Convert the arrays in the "Roll" parameters to tuples
        for i, params in enumerate(transformers_parameters["spectrogram"]["Roll"]):
            transformers_parameters["spectrogram"]["Roll"][i]["shift_dims"] = tuple(params["shift_dims"])
            transformers_parameters["spectrogram"]["Roll"][i]["dims"] = tuple(params["dims"])
    
    configs = {}

    if args.combination_mode == "all":
        configs = combine_all(transformers_parameters)
        pass
    elif args.combination_mode == "one_each":
        configs = combine_one_each(transformers_parameters)
    else:
        raise ValueError(f"Invalid combination mode: {args.combination_mode}")

    # Print the number of configs generated
    print(f"Generated {len(configs)} configurations")

    # If the number is larger than 50, confirm that the user wants to proceed
    if len(configs) > 50:
        print("Warning: the number of configurations generated is large. Continue? (y/n)")
        if input() != "y":
            exit()

    for key, config in configs.items():
        # Write the config to a JSON file
        with open(f"{args.output_dir}/{key}.json", "w") as f:
            json.dump(config, f, indent=4)