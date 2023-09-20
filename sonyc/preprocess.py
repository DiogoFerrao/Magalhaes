import os
import subprocess
from pathlib import Path
import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(prog="preprocess.py")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/media/magalhaes/sound/dataset_1673434256.csv",
        help="path to dataset csv file",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="cpu",
        help="directory used to store the embeddings",
    )
    args = parser.parse_args()

    dataset_df = pd.read_csv(args.dataset_path)

    dataset_files = list(dataset_df["audio_filename"])
    currents_files = [name.rstrip(".npz") for name in os.listdir(args.embeddings_dir)]
    dataset_filenames = []
    missing_files = []
    # unnecessary_files = currents_files.copy()

    for path in dataset_files:
        dataset_filenames.append(path.split("/")[-1].rstrip(".wav"))

    for i, file in enumerate(dataset_filenames):
        # unnecessary_files.pop(i)
        if file not in currents_files:
            missing_files.append(dataset_files[i])

    # out = subprocess.run(["rm"] + [f"{args.embeddings_dir}{name}.npz" for name in unnecessary_files])
    # print(out)

    out = subprocess.run(
        ["openl3", "audio"]
        + ([f"{str(Path(f))}" for f in missing_files])
        + f"--output-dir {args.embeddings_dir} --input-repr mel256 --content-type env --audio-embedding-size 512 --audio-hop-size 1.0 --audio-batch-size 16".split(
            " "
        )
    )
    # print(out)


if __name__ == "__main__":
    main()
