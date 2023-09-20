import os
import argparse

from vision_data_engine.utils.general import get_class_names


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Format labels annotated by Schreder using Label Studio",
    )
    parser.add_argument("labels_dir", help="directory with the labels to be moved")
    parser.add_argument("schreder_names", help="text file with class names")
    parser.add_argument(
        "--names",
        default="../../yolov7/data/schreder.names",
        help="text file with class names",
    )
    args = parser.parse_args()

    labels_dir = args.labels_dir

    schreder_classes = get_class_names(args.schreder_names)
    classes = get_class_names(args.names)

    for label in os.listdir(labels_dir):
        if label.endswith(".txt"):
            res_lines = []
            with open(os.path.join(labels_dir, label), "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == "":
                        continue
                    split = line.split(" ")
                    schreder_class_id = int(split[0])
                    class_id = classes.index(schreder_classes[schreder_class_id])
                    split[0] = str(class_id)
                    res_lines.append(" ".join(split) + "\n")

            with open(os.path.join(labels_dir, label), "w") as f:
                f.writelines(res_lines)


if __name__ == "__main__":
    main()
