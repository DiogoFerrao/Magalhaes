import json
import argparse
import os
from typing import Dict, List, Union

def get_last_part(filename: str) -> str:
    """Retrieve the last part of the filename after the last underscore."""
    return filename.split("_")[-1]

def get_dir_name(filename: str) -> str:
    """Retrieve the relevant part of the filename based on underscores."""
    temp = filename.split("_")
    transform_name = "_".join(temp[2:])
    return transform_name

def create_configuration_files(user_config: Dict[str, Union[str, int, float]], preprocessing_configs_dir: str, output_dir: str, experiment_name_prefix: str, class_imbalance_augment: bool) -> None:
    """Create configuration files based on user input."""
    
    dataset_csv_name = user_config["dataset_csv"].split("/")[-1].rstrip(".csv") # type: ignore

    files = os.listdir(preprocessing_configs_dir)
    filenames_without_extension = [os.path.splitext(file)[0] for file in files]

    if class_imbalance_augment:
        unique_names = set()
        for name in filenames_without_extension:
            parts = name.split("_")
            if len(parts) > 1:
                name_without_last = "_".join(parts[:-1])
                unique_names.add(name_without_last)
        filenames_without_extension = list(unique_names)


    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Wipe the directory if it exists
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    idx = 0
    for preprocessing_config in filenames_without_extension:
        last_part = get_last_part(preprocessing_config)
        dir_name = get_dir_name(preprocessing_config)

        new_config = user_config.copy()
        new_config["split_base"] = os.path.join(new_config["data_dir"], dir_name, dataset_csv_name) # type: ignore
        new_config["test_split"] = f"{new_config['split_base']}_split2.pkl"
        
        # Remove the trailing "last_part" from "dir_name"
        dir_name = dir_name.rstrip(f"_{last_part}")

        # Create the exp_name based on your requirements
        new_config["exp_name"] = f"{experiment_name_prefix}_{last_part}_{dir_name}"
        
        new_config["checkpoint"] = os.path.join(new_config['checkpoint_dir'], new_config['exp_name'], "model_best_1.pth") # type: ignore

        # Update the output filepath to remove the last part from the filename
        output_filepath = os.path.join(output_dir, f"{new_config['exp_name']}_config.json")
        
        with open(output_filepath, "w") as config_file:
            json.dump(new_config, config_file, indent=4)

        idx += 1
    
    print(f"Generated {idx} configuration files in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JSON configuration files for sound ablation experiments."
    )
    parser.add_argument(
        "--user_config",
        type=str,
        help="Path to the user JSON configuration file.",
        default="./config/sound_ablation_default_config.json"
    )
    parser.add_argument(
        "--preprocessing_configs_dir",
        type=str,
        default="./preprocessing_configs",
        help="Preprocessing configs (JSON) directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for generated configuration files.",
        default="./sound_ablation_configs"
    )
    parser.add_argument("--class_imbalance_augment", action="store_true", help="Whether to use class imbalance augmentation.")

    args = parser.parse_args()
    
    experiment_name_prefix = "Precomputed_Spec_MData"

    # Load user configuration from the provided JSON file
    with open(args.user_config, 'r') as file:
        user_config = json.load(file)

    create_configuration_files(user_config, args.preprocessing_configs_dir, args.output_dir, experiment_name_prefix, args.class_imbalance_augment)
