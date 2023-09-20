import argparse
import json
import os

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
        "input_dir",
        help="Directory with the json files corresponding to labels in labelme format",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to store the labels in Darknet/Yolo format",
    )
    parser.add_argument(
        "--labels",
        default="../../yolov7/data/schreder.names",
        help="File with label names",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    names = get_class_names(args.labels)

    for filename in os.listdir(args.input_dir):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(args.input_dir, filename)
        json_data = None
        with open(file_path) as fp:
            json_data = json.load(fp)

        lines = []
        for shape in json_data["shapes"]:
            if shape["shape_type"] != "rectangle":
                print(f"Invalid shape type at: {filename}")
                continue
            norm_points = shape["points"]

            norm_points[0][0] /= json_data["imageWidth"]
            norm_points[1][0] /= json_data["imageWidth"]
            norm_points[0][1] /= json_data["imageHeight"]
            norm_points[1][1] /= json_data["imageHeight"]

            norm_points[0][0] = clamp(norm_points[0][0], 0, 1)
            norm_points[0][1] = clamp(norm_points[0][1], 0, 1)
            norm_points[1][0] = clamp(norm_points[1][0], 0, 1)
            norm_points[1][1] = clamp(norm_points[1][1], 0, 1)

            try:
                id = names.index(shape["label"])
            except:
                id = names.index("other")

            x, y, w, h = convert_xyxy_to_xywh(norm_points)
            lines.append(f"{id} {x} {y} {w} {h}\n")

        with open(os.path.join(args.output_dir, f"{filename[:-5]}.txt"), "w") as fp:
            fp.writelines(lines)


if __name__ == "__main__":
    main()
