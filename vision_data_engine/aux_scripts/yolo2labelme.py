#!/usr/bin/env python

import argparse
import base64
import json
import os

from vision_data_engine.utils.general import get_class_names

import labelme
from labelme.utils import img_b64_to_arr


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="Directory with .txt outputted by Yolo")
    parser.add_argument(
        "output_dir", help="Directory to store the labels in labelme format (.json)"
    )
    parser.add_argument("image_dir", help="Directory with the images")
    parser.add_argument(
        "--names",
        default="../../yolov7/data/schreder.names",
        help="Text file defining class names",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    names = get_class_names(args.names)

    for filename in os.listdir(args.input_dir):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(args.input_dir, filename)
        img_path = os.path.join(args.image_dir, f"{filename[:-4]}.jpg")
        data = labelme.LabelFile.load_image_file(img_path)
        img_data = base64.b64encode(data).decode("utf-8")
        img_height, img_width, _ = img_b64_to_arr(img_data).shape

        data = dict(
            version="5.0.2",
            flags={},
            shapes=[],
            imagePath="",
            imageData=img_data,
            imageHeight=img_height,
            imageWidth=img_width,
        )

        for i, line in enumerate(open(file_path).readlines()):
            line_arr = line.split(" ")
            class_id = line_arr[0]
            x = float(line_arr[1])
            y = float(line_arr[2])
            width = float(line_arr[3])
            height = float(line_arr[4])
            x1 = (x - width / 2) * img_width
            x2 = (x + width / 2) * img_width
            y1 = (y - height / 2) * img_height
            y2 = (y + height / 2) * img_height

            box_data = dict(
                label=names[int(class_id)],
                points=[[x1, y1], [x2, y2]],
                group_id=None,
                shape_type="rectangle",
                flags={},
            )
            data["shapes"].append(box_data)

        with open(os.path.join(args.output_dir, f"{filename[:-4]}.json"), "w") as fp:
            json.dump(data, fp)


if __name__ == "__main__":
    main()
