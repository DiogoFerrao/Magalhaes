import argparse
import os
from pathlib import Path

from vision_utils.general import get_class_names


def main():
    """Remove images that don't contain any classes of interest"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Remove images that don't contain any classes of interest and creates a new filtered dataset",
    )
    parser.add_argument("dataset", help="txt of the dataset to filter")
    parser.add_argument(
        "--output_dir", default="./out", help="output directory for the new txt file"
    )
    parser.add_argument(
        "--labels", default="../../yolov7/data/schreder.names", help="file with labels"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_dataset_path = Path(args.output_dir) / (
        Path(args.dataset).stem + "_filtered.txt"
    )

    # Other class name
    OTHER_CLASS_NAME = "other"

    # Classes of interest
    names = get_class_names(args.labels)
    non_existant = 0
    filtered_images = []
    for line in open(args.dataset).readlines():
        line = line.strip()
        if not line:
            continue
        label_file = line.replace("/images/", "/labels/").replace(".jpg", ".txt")
        label_file = Path(args.dataset).parent.joinpath(label_file)
        if not label_file.exists():
            non_existant += 1
            continue
        for bbox in open(label_file).readlines():
            bbox = bbox.strip()
            if not bbox:
                continue
            class_name = names[int(bbox.split()[0])]
            if class_name != OTHER_CLASS_NAME:
                filtered_images.append(line)
                break
    print(f"Non existent: {non_existant}")
    with open(out_dataset_path, "w") as f:
        f.writelines("\n".join(filtered_images))


if __name__ == "__main__":
    main()
