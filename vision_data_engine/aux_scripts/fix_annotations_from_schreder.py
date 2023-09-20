import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fix label duplication found in data annotated by Schreder using Label Studio",
    )
    parser.add_argument("labels_dir", help="directory with the labels to be moved")

    args = parser.parse_args()

    labels_dir = args.labels_dir

    for label in os.listdir(labels_dir):
        if label.endswith(".txt"):
            res_lines = []
            with open(os.path.join(labels_dir, label), "r") as f:
                for line in f.readlines():
                    if line not in res_lines:
                        res_lines.append(line)

            with open(os.path.join(labels_dir, label), "w") as f:
                f.writelines(res_lines)


if __name__ == "__main__":
    main()
