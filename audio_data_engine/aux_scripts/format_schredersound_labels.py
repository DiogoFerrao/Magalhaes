import argparse
from datetime import datetime, timedelta
import os

import pandas as pd


def find_associated_file(path, time):
    for filename in os.listdir(path):
        if filename.endswith(".wav"):
            # Fix time gap between consecutive audio clips
            epsilon = timedelta(seconds=0.5)
            file_time_str = filename[:-4].replace("_", ":")
            file_time = datetime.strptime(file_time_str, "%Y-%m-%d %H:%M:%S.%f%z")
            between = (
                file_time - epsilon - timedelta(seconds=10) < time < file_time + epsilon
            )
            if between:
                return filename
    return None


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Format annotations made by schreder to audios to a csv file",
    )
    parser.add_argument(
        "--annotations",
        default="/media/magalhaes/schreder_sound/22_9_2022/ground_truth/ground_truth.csv",
        help="csv file with annotations from schreder",
    )
    parser.add_argument(
        "--labels",
        default="../../rethink/data/schreder.names",
        help="file with expected labels",
    )
    parser.add_argument(
        "--sound_dir",
        default="/media/magalhaes/schreder_sound/22_9_2022",
        help="directory with the audios",
    )
    parser.add_argument(
        "--output_dir",
        default="/media/magalhaes/schreder_sound",
        help="directory to store the resulting csv file",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sound_dir.endswith("/"):
        args.sound_dir = args.sound_dir.rstrip("/")
    dataset_name = os.path.basename(args.sound_dir)

    annotation_fine_to_label_dict = {
        "Car": "car",
        "Bus": "bus",
        "Truck": "truck",
        "Motorbike": "motorcycle",
        "Bicycle": "bicycle",
        "Horn": "horn",
    }

    annotation_coarse_to_label_dict = {
        "People": "person",
        "Siren": "siren",
    }

    label_to_id = {}
    n_lost_labels = 0
    total_labels = 0

    for i, label in enumerate(open(args.labels).readlines()):
        label_to_id[label.strip()] = i

    columns = "audio_filename,dataset,latitude,longitude,year,week,day,hour"
    columns_arr = columns.split(",")
    labels_arr = []

    for key in label_to_id.keys():
        columns_arr.append(key)
        labels_arr.append(key)

    data = dict.fromkeys(columns_arr, [])
    df = pd.DataFrame(data)
    lines = []
    with open(args.annotations) as fp:
        lines = fp.read().split("\n")

    audio_file_entries = {}
    # read annotations
    for i, line in enumerate(lines):
        if line.strip() == "":
            continue
        direction, coarse, fine, time = line.split(",")

        try:
            if coarse in ["Light", "Heavy", ""]:
                cls = annotation_fine_to_label_dict[fine]
            else:
                cls = annotation_coarse_to_label_dict[coarse]
        except:
            print(f"Found unexpected annotation: {coarse}, {fine}")
            continue

        date_time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f%z")
        source_filename = find_associated_file(args.sound_dir, date_time)

        total_labels += 1
        if source_filename is None:
            n_lost_labels += 1
            continue

        # first annotation corresponding to a certain source file
        if source_filename not in audio_file_entries:
            audio_file_entries[source_filename] = dict(
                audio_filename=os.path.abspath(
                    os.path.join(args.sound_dir, source_filename)
                ),
                dataset=dataset_name,
                latitude="",
                longitude="",
                year=date_time.year,
                week=date_time.strftime("%V"),
                day=date_time.day,
                hour=date_time.hour,
            )
            for label in labels_arr:
                audio_file_entries[source_filename][label] = 1 if cls == label else 0
        else:
            for label in labels_arr:
                if cls == label:
                    audio_file_entries[source_filename][label] = 1

    out_path = os.path.join(args.output_dir, f"{dataset_name}_2_labels.csv")
    df = pd.concat(
        (df, pd.DataFrame(list(audio_file_entries.values()))), ignore_index=True
    )
    df.to_csv(out_path, index=False)

    print(
        f"Found {n_lost_labels} out of {total_labels} annotations without any audio clip associated"
    )
    print(f"Writing formatted dataset to: {out_path}")


if __name__ == "__main__":
    main()
