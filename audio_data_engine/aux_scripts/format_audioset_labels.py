import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # fmt: off
    parser.add_argument("annotations", help="csv file with Audioset annotations")
    parser.add_argument("sounds_dir", help="directory with Audioset audios")
    parser.add_argument("--labels", default="../../rethink/data/schreder.names", help="file with expected labels")
    parser.add_argument("--output_dir", default="/home/guests2/msg/Audioset/audioset-processing/data", help="directory to store the resulting csv file with the new format")
    # fmt: on

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_df = pd.read_csv(args.annotations, names=["yt_id", "start", "end", "mid"])
    formatted_dataset_data = {"filepath": [], "label_id": []}

    for row in dataset_df.itertuples():
        filename = f"{row.yt_id}_{int(row.start)}.wav"
        filepath = os.path.join(args.sounds_dir, filename)
        if os.path.isfile(filepath):
            formatted_dataset_data["filepath"].append(filepath)
            formatted_dataset_data["label_id"].append(row.mid)

    formatted_dataset_df = pd.DataFrame(data=formatted_dataset_data)
    out_filepath = os.path.join(args.output_dir, "audioset.csv")
    print(f"Writing formatted dataset to: {out_filepath}")
    formatted_dataset_df.to_csv(out_filepath, index=False)


if __name__ == "__main__":
    main()
