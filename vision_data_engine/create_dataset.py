import argparse
from pathlib import Path
import os
import time

import numpy as np
import pandas as pd


def image_bb_samples_per_class(image_path: str, classes: list[str]):
    """Returns a dictionary with the number of samples per class in the image"""
    res = dict(zip(classes, [0 for _ in range(len(classes))]))
    label_path = image_path.replace("images/", "labels/").replace(
        image_path.split(".")[-1], "txt"
    )
    if os.path.isfile(label_path):
        for line in open(label_path).readlines():
            if line.strip() == "":
                continue
            cls_id = int(line.split()[0])
            res[classes[cls_id]] += 1
    return res


def main():
    # Split a set of datasets into 50/50 and create a test set if it doesn't exist
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=""
    )
    # fmt: 0ff
    parser.add_argument(
        "--datasets_file",
        default="../yolov7/data/all.dataset",
        help="txt file combining datasets",
    )
    parser.add_argument(
        "--output_dir",
        default="/media/magalhaes/vision/datasets",
        help="output labels directory",
    )
    parser.add_argument("--test_set", default=None, help="txt file with test set")
    parser.add_argument(
        "--create_test_set",
        action="store_true",
        help="Creates a new test set and save the old one",
    )
    parser.add_argument(
        "--names",
        default="../yolov7/data/schreder.names",
        help="File with class names",
    )
    parser.add_argument(
        "--test_set_size", default=500, type=int, help="Size of test set"
    )
    parser.add_argument(
        "--max_class_ratio",
        default=0.2,
        type=float,
        help="Maximum ratio of samples per class present in the test set",
    )
    parser.add_argument(
        "--test_subsets",
        nargs="+",
        default=[
            "/media/magalhaes/schreder/day_test_set/schreder_day.txt",
            "/media/magalhaes/schreder/night_test_set/schreder_night.txt",
        ],
        help="Maximum ratio of samples per class present in the test set",
    )
    # fmt: on

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    curr_time = int(time.time())

    final_dataset_data = {"filepath": [], "dataset": []}
    classes = []
    datasets = []

    # Add classes to dataframe columns
    for i, line in enumerate(open(args.names).readlines()):
        final_dataset_data[line.strip()] = []
        classes.append(line.strip())

    with open(Path(args.datasets_file)) as fp:
        datasets = fp.read().split("\n")

    # Read datasets and store into the dataframe
    for dataset in datasets:
        if dataset.strip() == "":
            continue
        curr_dataset_path = Path(dataset)
        curr_dataset_dir_path = os.path.dirname(curr_dataset_path)

        with open(curr_dataset_path) as fp:
            dataset_lines = fp.read().split("\n")
            for filename in dataset_lines:
                if filename.strip() == "":
                    continue
                image_filepath = str(
                    os.path.abspath(os.path.join(curr_dataset_dir_path, filename))
                )
                image_class_bb = image_bb_samples_per_class(image_filepath, classes)

                final_dataset_data["filepath"].append(image_filepath)
                final_dataset_data["dataset"].append(dataset)
                for cls in classes:
                    final_dataset_data[cls].append(image_class_bb[cls])

    train_dataset_df = pd.DataFrame(data=final_dataset_data)

    # Create a test set from images from Schreder
    # We start sampling from underrepresneted classes
    test_set = None
    if args.create_test_set:
        test_set = []
        schreder_subset = train_dataset_df[
            train_dataset_df["filepath"].str.contains("schreder")
        ]
        samples_per_class = {}

        # Calculate number of samples per class
        curr_test_set_size = 0
        while curr_test_set_size < args.test_set_size:
            for cls in classes:
                if cls not in samples_per_class.keys():
                    samples_per_class[cls] = 0

                cls_subset = schreder_subset[schreder_subset[cls] != 0]
                n_images_with_class = cls_subset[cls].sum()

                if samples_per_class[cls] < n_images_with_class * args.max_class_ratio:
                    samples_per_class[cls] += 1
                    curr_test_set_size += 1

        # Sample images per class according to the number of samples per class
        for cls in samples_per_class.keys():
            cls_subset = schreder_subset[schreder_subset[cls] != 0]
            chosen_indexes = np.random.choice(
                cls_subset.index, size=(samples_per_class[cls],), replace=False
            )
            for i in chosen_indexes:
                filepath = str(train_dataset_df.loc[[i]].filepath.values[0])
                test_set.append(filepath)
                schreder_subset = schreder_subset.drop(index=i)
                train_dataset_df = train_dataset_df.drop(index=i)
        train_dataset_df = train_dataset_df.reset_index(drop=True)

        # Add test subsets (for example, night and day)
        for test_subset in args.test_subsets:
            for line in open(test_subset).readlines():
                filepath = line.strip()
                if filepath == "":
                    continue
                if filepath.startswith("."):
                    filepath = os.path.abspath(
                        os.path.join(os.path.dirname(test_subset), filepath)
                    )
                test_set.append(filepath)
        # Save new test set
        test_set_path = os.path.join(args.output_dir, "vision_test_set.txt")
        old_test_set_path = os.path.join(
            args.output_dir, f"vision_test_set_{int(time.time())}.txt"
        )

        # Save old test set
        if os.path.exists(test_set_path):
            with open(old_test_set_path, "w") as fp:
                with open(test_set_path, "r") as fp_old:
                    for line in fp_old.readlines():
                        fp.write(line)

        # Save new test set
        with open(test_set_path, "w") as fp:
            for out_line in test_set:
                fp.write(out_line + "\n")
        print(f"New test set created: {test_set_path}")
    else:
        # Test set is provided, so we just remove the samples present in the test set from the train dataset
        if args.test_set is None:
            test_set_path = os.path.join(args.output_dir, "vision_test_set.txt")
            print(f"No test set was provided, using default test set: {test_set_path}")
        else:
            test_set_path = args.test_set
            print(f"Using test set: {test_set_path}")
        test_set = [
            line.strip()
            for line in open(test_set_path).readlines()
            if line.strip() != ""
        ]
        for i, row in enumerate(train_dataset_df.itertuples()):
            if row.filepath in test_set:
                train_dataset_df = train_dataset_df.drop(index=i)
        train_dataset_df = train_dataset_df.reset_index(drop=True)

    complete_dataset = train_dataset_df.filepath.values.tolist()

    # Sample splits (stratified)
    # Again we start by sampling images with rare classes to guarantee that they are represented in both splits
    splits = [[], []]
    for dataset in datasets:
        dataset_subset = train_dataset_df[train_dataset_df["dataset"] == dataset]
        dataset_subset = dataset_subset.sample(frac=1)
        for cls in dataset_subset.iloc[:, 2:].sum().sort_values().keys():
            cls_subset = dataset_subset[dataset_subset[cls] != 0]
            splits[0].extend(cls_subset.filepath.values[: len(cls_subset) // 2])
            splits[1].extend(cls_subset.filepath.values[len(cls_subset) // 2 :])
            # remove files
            merged = pd.merge(dataset_subset, cls_subset, how="outer", indicator=True)
            dataset_subset = merged.loc[merged["_merge"] == "left_only"].drop(
                columns="_merge"
            )

    # Save dataset files
    dataset_path = os.path.join(args.output_dir, f"vision_{curr_time}.txt")
    with open(dataset_path, "w") as fp:
        for out_line in complete_dataset:
            fp.write(out_line + "\n")

    print(f"New dataset created: {dataset_path}")

    for i, split in enumerate(splits):
        with open(
            os.path.join(args.output_dir, f"vision_{curr_time}_split_{i+1}.txt"), "w"
        ) as fp:
            for out_line in split:
                fp.write(out_line + "\n")


if __name__ == "__main__":
    main()
