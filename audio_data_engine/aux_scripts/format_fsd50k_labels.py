import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--annotations",
        default="/media/magalhaes/FSD50K/dev.csv",
        help="csv file with FSD50K labels",
    )
    parser.add_argument(
        "--sounds_dir",
        default="/media/magalhaes/FSD50K/dev_audio",
        help="directory with FSD50K audios",
    )
    parser.add_argument(
        "--labels",
        default="../../rethink/data/schreder.names",
        help="file with expected labels",
    )
    parser.add_argument(
        "--output_dir",
        default="/media/magalhaes/FSD50K",
        help="directory to store the resulting csv file with the new format",
    )
    parser.add_argument(
        "--labels_translation",
        default="../data/fsd50k2schreder.names",
        help="label names translation file",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    label_to_id = {}
    fsd50k_to_schreder = {}

    # label -> id, car -> 1
    for i, label in enumerate(open(args.labels).readlines()):
        label_to_id[label.strip()] = i

    # translate fsd50k labels to our labels
    for line in open(args.labels_translation).readlines():
        fsd50k_label, schreder_label = line.split(",")
        fsd50k_to_schreder[fsd50k_label] = schreder_label.strip()

    # construct dataframe
    columns = "audio_filename,dataset,latitude,longitude,year,week,day,hour"
    columns_arr = columns.split(",")
    labels_arr = []

    for key in label_to_id.keys():
        columns_arr.append(key)
        labels_arr.append(key)

    data = dict.fromkeys(columns_arr, [])
    out_df = pd.DataFrame(data)

    in_data = pd.read_csv(args.annotations)
    out_data = []

    # Stats
    n_interesting_audios = 0
    n_total_audios = 0
    labels_counters = dict.fromkeys(labels_arr, 0)

    # read annotations
    for i, row in in_data.iterrows():
        source_filename = str(row["fname"])
        fsd50k_labels = row["labels"].split(",")
        schreder_labels = [
            fsd50k_to_schreder[label]
            for label in fsd50k_labels
            if label in fsd50k_to_schreder.keys()
        ]

        n_total_audios += 1
        if len(schreder_labels) > 0:
            n_interesting_audios += 1

            out_data.append(
                dict(
                    audio_filename=os.path.abspath(
                        os.path.join(args.sounds_dir, f"{source_filename}.wav")
                    ),
                    dataset="FSD50k",
                    latitude="",
                    longitude="",
                    year="",
                    week="",
                    day="",
                    hour="",
                )
            )
            # stats
            for label in schreder_labels:
                labels_counters[label] += 1

            for label in labels_arr:
                out_data[-1][label] = 1 if label in schreder_labels else 0

    out_path = os.path.join(args.output_dir, "schreder_labels.csv")
    out_df = pd.concat((out_df, pd.DataFrame(out_data)), ignore_index=True)
    out_df.to_csv(out_path, index=False)

    print(
        f"Found {n_interesting_audios} audio files in FSD50K with classes of interest out of a total of {n_total_audios}. "
    )
    print("Number of examples found per class:")
    print(labels_counters)
    print(f"Writing formatted dataset to: {out_path}")


if __name__ == "__main__":
    main()
