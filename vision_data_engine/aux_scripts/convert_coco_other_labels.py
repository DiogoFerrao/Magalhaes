import argparse
import os


def main():
    """Convert unwanted labels to "other" """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", default=None, help="labels directory")
    parser.add_argument(
        "--output_dir", default=None, help="output directory with new labels"
    )
    parser.add_argument(
        "--labels", default="../../yolov7/data/schreder.names", help="file with labels"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    n_labels = 0
    other_label_id = 0
    emergency_vehicle_id = 0

    for i, line in enumerate(open(args.labels).readlines()):
        if line.strip() != "":
            n_labels += 1
        if line.strip() == "emergency_vehicle":
            emergency_vehicle_id = i
        if line.strip() == "other":
            other_label_id = i

    for label_file in os.listdir(args.input_dir):
        in_label_file_path = os.path.join(args.input_dir, label_file)
        out_label_file_path = os.path.join(args.output_dir, label_file)

        new_targets = []
        for i, line in enumerate(open(in_label_file_path).readlines()):
            target = line.split(" ")
            cls = int(target[0])
            if cls == emergency_vehicle_id or cls >= n_labels:
                target[0] = str(other_label_id)
            new_targets.append(" ".join(target))

        with open(out_label_file_path, "w") as fp:
            fp.writelines(new_targets)


if __name__ == "__main__":
    main()
