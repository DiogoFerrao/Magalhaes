import argparse
import os

import cv2

from vision_data_engine.utils.general import get_class_names


def convert_xyxy_to_xywh(xyxy):
    # convert from top-left, bottom-right box to center, width, height box
    x = (xyxy[0][0] + xyxy[1][0]) / 2
    y = (xyxy[0][1] + xyxy[1][1]) / 2
    w = abs(xyxy[0][0] - xyxy[1][0])
    h = abs(xyxy[0][1] - xyxy[1][1])
    return x, y, w, h


def clamp(x, min_x, max_x):
    return max(min_x, min(max_x, x))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir", default=None, help="input exdark labels directory"
    )
    parser.add_argument("--output_dir", default=None, help="output labels directory")
    parser.add_argument(
        "--labels", default="../../yolov7/data/schreder.names", help="labels file"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    labels = get_class_names(args.labels)

    for filename in os.listdir(args.input_dir):
        if not filename.endswith(".txt"):
            continue

        lines = []
        with open(os.path.join(args.input_dir, filename)) as fp:
            lines = fp.read().split("\n")

        # Get image size
        im = cv2.imread(os.path.join(args.input_dir, "..", "images", filename[:-4]))
        if im is None:
            continue
        print(filename)
        image_height = im.shape[0]
        image_width = im.shape[1]

        # ignore first line
        lines = lines[1:]
        out_lines = []
        for line in lines:
            values = line.split(" ")
            if len(values) < 5:
                continue
            label = values[0].lower()

            if label == "people":
                label = "person"
            elif label == "motorbike":
                label = "motorcycle"
            if label not in labels:
                continue
            id = labels.index(label)

            x = (float(values[1]) + float(values[3]) / 2) / image_width
            y = (float(values[2]) + float(values[4]) / 2) / image_height
            w = float(values[3]) / image_width
            h = float(values[4]) / image_height
            out_lines.append(f"{id} {x} {y} {w} {h}\n")

        with open(os.path.join(args.output_dir, f"{filename[:-8]}.txt"), "w") as fp:
            fp.writelines(out_lines)


if __name__ == "__main__":
    main()
