import json
import argparse
import os

def get_dir_name(filename: str):
    temp = preprocessing_config.split("_")
    transform_name = "_".join(temp[2:])
    return transform_name

if __name__ == "__main__":
    # Receive user input from the command line
    parser = argparse.ArgumentParser(
        description="Generate JSON configuration files for sound ablation experiments."
    )
    parser.add_argument(
        "--preprocessing_configs_dir",
        type=str,
        default="../preprocessing_configs",
        help="Preprocessing configs (JSON) directory.",
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
        "--dataset_csv",
        default="/media/magalhaes/sound/datasets/sound_1695725868.csv",
        help="Dataset CSV file.",
    )
    parser.add_argument("--split", type=int, default=1, help="Split number.")
    parser.add_argument("--full_train", type=bool, default=False, help="Full training.")
    parser.add_argument(
        "--precomputed_spec", type=bool, default=True, help="Precomputed spectrogram."
    )


    args = parser.parse_args()


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
        "dataset_csv": args.dataset_csv,
        "split": args.split,
        "full_train": args.full_train,
        "precomputed_spec": args.precomputed_spec,
        "spec_transforms": [],
        "waveform_transforms": [],
    }
    
    experiment_name_prefix = "Precomputed_Spec_MData_P04"

    # consider changing the prefix based on the name of the config file
    # e.g. Precomputed_Spec_MData_Top7_1

    dataset_csv_name = args.dataset_csv.split("/")[-1].rstrip(".csv")

    files = os.listdir(args.preprocessing_configs_dir)

    filenames_without_extension = [os.path.splitext(file)[0] for file in files]


    idx = 0
    # For every file in the preprocessing configs directory
    for preprocessing_config in filenames_without_extension:

        dir_name = get_dir_name(preprocessing_config)
        print(dir_name)

        new_config = user_config.copy()
        new_config["split_base"] = new_config["data_dir"] +"/"+ dir_name + "/" + dataset_csv_name
        new_config["test_split"] = new_config["split_base"] + "_split2.pkl"

        new_config["exp_name"] = f"{experiment_name_prefix}_{dir_name}"
        new_config["checkpoint"] = (
            f"{new_config['checkpoint_dir']}/{new_config['exp_name']}/model_best_1.pth"
        )

        # Create a new configuration JSON file
        with open(
                f"{experiment_name_prefix}_{dir_name}_config.json", "w"
        ) as config_file:
            json.dump(new_config, config_file, indent=4)

        idx += 1
    
    print(f"Generated {idx} configuration files.")