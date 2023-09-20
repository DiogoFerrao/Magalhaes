import argparse
import os
import subprocess

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Remove audios that are not in the dataset",
    )
    parser.add_argument("sound_dir", help="directory with the audios to be removed")
    parser.add_argument(
        "dataset_csv",
        help="csv specifying the dataset",
    )

    args = parser.parse_args()

    dataset_df = pd.read_csv(args.dataset_csv)
    paths_to_remove = []

    for path in os.listdir(args.sound_dir):
        abs_path = os.path.join(args.sound_dir, path)
        if path.endswith(".wav"):
            if abs_path not in dataset_df.values:
                paths_to_remove.append(abs_path)
    subprocess.call(["rm"] + paths_to_remove)


if __name__ == "__main__":
    main()
