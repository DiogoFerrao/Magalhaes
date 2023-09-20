import argparse
import os
import time
import numpy.random as np_rand
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create a dataset and splits for cross validation",
    )
    parser.add_argument(
        "--datasets",
        default="../rethink/data/all.dataset",
        help="text file with the paths to all csv files to be used by the final dataset",
    )
    parser.add_argument(
        "--labels",
        default="../rethink/data/schreder.names",
        help="file with expected labels",
    )
    parser.add_argument(
        "--output_dir",
        default="/media/magalhaes/sound/datasets",
        help="directory to store the resulting csv file",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_test_ratio = [0.5, 0.5]

    finalDF = pd.DataFrame()
    for dataset_path in open(args.datasets).readlines():
        df = pd.read_csv(dataset_path.strip("\n"))
        df = df.drop(
            ["latitude", "longitude", "year", "week", "day", "hour"],
            axis=1,
            errors="ignore",
        )

        df["split"] = np_rand.choice(
            ["split1", "split2"], len(df.index), p=train_test_ratio
        )

        finalDF = pd.concat((finalDF, df), ignore_index=True)

    out_path = os.path.join(args.output_dir, f"sound_{int(time.time())}.csv")
    finalDF.to_csv(out_path)
    print(f"Created dataset at: {out_path}")


if __name__ == "__main__":
    main()
