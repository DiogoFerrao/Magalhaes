import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--annotations",
        default="/media/magalhaes/ESC-50/esc50.csv",
        help="csv file with ESC-50 labels",
    )
    parser.add_argument(
        "--sounds_dir",
        default="/media/magalhaes/ESC-50/audio",
        help="directory with ESC-50 audios",
    )
    parser.add_argument(
        "--labels",
        default="../../rethink/data/schreder.names",
        help="file with expected labels",
    )
    parser.add_argument(
        "--output_dir",
        default="/media/magalhaes/ESC-50",
        help="directory to store the resulting csv file with the new format",
    )
    parser.add_argument(
        "--labels_translation",
        default="../data/esc502schreder.names",
        help="label names translation file",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    label_to_id = {}
    esc50_to_schreder = {}

    # label -> id, car -> 1
    for i, label in enumerate(open(args.labels).readlines()):
        label_to_id[label.strip()] = i

    # translate fsd50k labels to our labels
    for line in open(args.labels_translation).readlines():
        esc50_label, schreder_label = line.split(",")
        esc50_to_schreder[esc50_label] = schreder_label.strip()

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
        source_filename = str(row["filename"])
        esc50_label = row["category"]
        schreder_label = (
            esc50_to_schreder[esc50_label]
            if esc50_label in esc50_to_schreder.keys()
            else ""
        )

        n_total_audios += 1
        if len(schreder_label) > 0:
            n_interesting_audios += 1

            out_data.append(
                dict(
                    audio_filename=os.path.abspath(
                        os.path.join(args.sounds_dir, source_filename)
                    ),
                    dataset="ESC50",
                    latitude="",
                    longitude="",
                    year="",
                    week="",
                    day="",
                    hour="",
                )
            )
            # stats
            labels_counters[schreder_label] += 1

            for label in labels_arr:
                out_data[-1][label] = 1 if label == schreder_label else 0

    out_path = os.path.join(args.output_dir, "schreder_labels.csv")
    out_df = pd.concat((out_df, pd.DataFrame(out_data)), ignore_index=True)
    out_df.to_csv(out_path, index=False)

    print(
        f"Found {n_interesting_audios} audio files in ESC-50 with classes of interest out of a total of {n_total_audios}. "
    )
    print("Number of examples found per class:")
    print(labels_counters)
    print(f"Writing formatted dataset to: {out_path}")


if __name__ == "__main__":
    main()
