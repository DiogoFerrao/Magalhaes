import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ground_truth_csv",
        default="/media/magalhaes/DCASE2017/training_set.csv",
        help="csv file with DCASE2017 labels",
    )
    parser.add_argument(
        "--sounds_dir",
        default="/media/magalhaes/DCASE2017/training_set",
        help="directory with FSD50K audios",
    )
    parser.add_argument(
        "--audioset-names", default="../../rethink/data/audioset.names", help=""
    )

    args = parser.parse_args()

    if "training_set.csv" in args.ground_truth_csv:
        dataset_df = pd.read_csv(
            args.ground_truth_csv,
            names=["yt_id", "start_time", "end_time", "labels", "labels_ids"],
        )
        names_path = "../data/dcase2017.names"

        # label_to_label_id = {}
        # audioset_df = pd.read_csv(args.audioset_names)

        # for i, row in audioset_df.iterrows():
        #     names = row["display_name"]
        #     for name in names.split(", "):
        #         label_to_label_id[name] = row["mid"]

        res_labels = []
        data = {"filepath": [], "label_id": []}
        for i, row in dataset_df.iterrows():
            # Save labels
            labels = row["labels_ids"].split(",")
            n_labels = []
            for label in labels:
                label = label.lstrip(" ")
                n_labels.append(label)
                if label not in res_labels:
                    res_labels.append(label)
            yt_id, start_time, end_time = (
                row["yt_id"],
                row["start_time"],
                row["end_time"],
            )
            filepath = f"Y{yt_id}_{start_time:.3f}_{end_time:.3f}.wav"
            data["filepath"].append(os.path.join(args.sounds_dir, filepath))
            data["label_id"].append(",".join(n_labels))

            # TODO: maybe change sample rate although this is problematic if in the future we decide to change it

        # Write new csv to save audio files path
        new_csv_path = os.sep.join(
            args.ground_truth_csv.split(os.sep)[:-1]
            + ["schreder_distil_" + args.ground_truth_csv.split(os.sep)[-1]]
        )
        pd.DataFrame(data).to_csv(new_csv_path, index=False)

        with open(names_path, "w") as fp:
            res_labels = list(map(lambda x: x + "\n", res_labels))
            fp.writelines(res_labels)

        print(f"Found {len(res_labels)} labels")
        print(f"Saving labels names to {names_path}")
        print(f"Saving new csv to {new_csv_path}")


if __name__ == "__main__":
    main()
