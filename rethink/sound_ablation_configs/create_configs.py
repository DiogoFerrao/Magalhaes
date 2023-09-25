import json
import argparse
from itertools import chain, combinations

# List of available transformations (you can add more if needed)
spec_transforms = ["Roll", "TimeMasking", "FreqMasking"]
waveform_transforms = [
    "AirAbsorption",
    "ClippingDistortion",
    "Gain",
    "GaussianNoise",
    "PitchShift",
    "TimeStretch",
    "LowPassFilter",
]

# Experiment name prefix
experiment_name_prefix = "sound_ablation_resnet_tiny"


def generate_combinations(transformations_list, combination_method="all"):
    all_combinations = []

    if combination_method == "all":
        # Generate all combinations of transformations
        for r in range(0, len(transformations_list) + 1):
            for combo in combinations(transformations_list, r):
                all_combinations.append(list(combo))
    elif combination_method == "one":
        # Generate only one combination per transformation
        for transformation in transformations_list:
            all_combinations.append([transformation])
        # Add the base configuration (no transformations)
        all_combinations.append([])
    elif combination_method == "custom":
        # Implement your custom combination rules here
        pass
    else:
        raise ValueError("Invalid combination method")

    return all_combinations


def generate_configurations(combinations_list, user_config):
    # Iterate through all combinations
    for idx, transformation_list in enumerate(combinations_list):
        # Create a new configuration dictionary with default values
        config = user_config.copy()

        # Update the configuration with user-provided values
        if user_config:
            config.update(user_config)

        # Update spec_transforms and waveform_transforms based on the current combination
        config["spec_transforms"] = [
            t for t in transformation_list if t in spec_transforms
        ]
        config["waveform_transforms"] = [
            t for t in transformation_list if t in waveform_transforms
        ]

        if len(transformation_list) == 0:
            config["exp_name"] = f"{experiment_name_prefix}_base"
        else:
            config["exp_name"] = (
                f"{experiment_name_prefix}_{'_'.join(transformation_list)}"
            )

        config["checkpoint"] = (
            f"{config['checkpoint_dir']}/{config['exp_name']}/model_best_1.pth"
        )

        # Create a new configuration JSON file
        with open(
            f"{experiment_name_prefix}_{idx + 1}_config.json", "w"
        ) as config_file:
            json.dump(config, config_file, indent=4)

    print(f"{len(combinations_list)} JSON configuration files generated.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON configuration files for different combinations of transformations."
    )
    parser.add_argument(
        "--combination_method",
        default="one",
        choices=["all", "one", "custom"],
        help="Combination method.",
    )
    parser.add_argument(
        "--data_dir",
        default="/media/magalhaes/sound/spectograms",
        help="Data directory.",
    )
    parser.add_argument("--dataaug", type=bool, default=True, help="Data augmentation.")
    parser.add_argument(
        "--pretrained", type=bool, default=True, help="Use pretrained model."
    )
    parser.add_argument("--scheduler", type=bool, default=False, help="Use scheduler.")
    parser.add_argument("--model", default="yolov7_tiny", help="Model name.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers.")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-3, help="Weight decay."
    )
    parser.add_argument("--mixup", type=float, default=0.0, help="Mixup factor.")
    parser.add_argument(
        "--self_distillation", type=bool, default=False, help="Self distillation."
    )
    parser.add_argument("--num_folds", type=int, default=2, help="Number of folds.")
    parser.add_argument(
        "--pretrained_weights",
        default="/media/magalhaes/sound/pretrained/yolov7-tiny-finetuned.pt",
        help="Pretrained weights path.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="/media/cache/magalhaes/sound/checkpoints",
        help="Checkpoint directory.",
    )
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes.")
    parser.add_argument(
        "--conf_threshold", type=float, default=0.5, help="Confidence threshold."
    )
    parser.add_argument("--names", default="./data/schreder.names", help="Names file.")
    parser.add_argument("--device", default="cuda:0", help="Device.")
    parser.add_argument(
        "--split_base",
        default="/media/magalhaes/sound/spectograms/sound_1686042151",
        help="Split base directory.",
    )
    parser.add_argument(
        "--test_split",
        default="/media/magalhaes/sound/spectograms/sound_1686042151_split2.pkl",
        help="Test split path.",
    )
    parser.add_argument(
        "--dataset_csv",
        default="/media/magalhaes/sound/datasets/sound_1686042151.csv",
        help="Dataset CSV file.",
    )
    parser.add_argument("--split", type=int, default=1, help="Split number.")
    parser.add_argument("--full_train", type=bool, default=False, help="Full training.")
    parser.add_argument(
        "--precomputed_spec", type=bool, default=False, help="Precomputed spectrogram."
    )
    parser.add_argument(
        "--from_waveform", type=bool, default=True, help="Load data from waveform."
    )

    args = parser.parse_args()

    # Define the list of transformations to use
    all_transforms = spec_transforms + waveform_transforms

    # Generate combinations of transformations
    combinations_list = generate_combinations(
        all_transforms, combination_method=args.combination_method
    )

    # Create a user_config dictionary containing only the specified command-line arguments
    user_config = {
        "data_dir": args.data_dir,
        "dataaug": args.dataaug,
        "pretrained": args.pretrained,
        "scheduler": args.scheduler,
        "model": args.model,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "sample_rate": args.sample_rate,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "mixup": args.mixup,
        "self_distillation": args.self_distillation,
        "num_folds": args.num_folds,
        "pretrained_weights": args.pretrained_weights,
        "checkpoint_dir": args.checkpoint_dir,
        "num_classes": args.num_classes,
        "conf_threshold": args.conf_threshold,
        "names": args.names,
        "device": args.device,
        "split_base": args.split_base,
        "test_split": args.test_split,
        "dataset_csv": args.dataset_csv,
        "split": args.split,
        "full_train": args.full_train,
        "precomputed_spec": args.precomputed_spec,
        "from_waveform": args.from_waveform,
    }

    # Generate configurations based on the combinations and user-provided values
    generate_configurations(combinations_list, user_config)


if __name__ == "__main__":
    main()
