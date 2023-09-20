import json
import os

import cv2
import pandas as pd


def clamp(x, min_x, max_x):
    return max(min_x, min(max_x, x))


def compute_dataset_stats(labels_path, dataset_txt):
    labels_counters = {}
    labels_area_counters = {}
    id_to_label_dict = {}

    if labels_path is not None:
        for i, line in enumerate(open(labels_path).readlines()):
            if line.strip() != "":
                labels_counters[line.strip()] = 0
                labels_area_counters[line.strip()] = 0
                id_to_label_dict[i] = line.strip()

    image_files = None
    with open(dataset_txt) as fp:
        image_files = fp.read().split()
        image_files = list(filter(lambda x: x.strip() != "", image_files))

    # COCO
    label_files = []
    for file_path in image_files:
        if "coco" in file_path:
            label_files.append(
                os.path.join(
                    os.sep.join(file_path.split("/")[:-3]),
                    "labels",
                    file_path.split("/")[-2],
                    os.path.basename(file_path)[:-3] + "txt",
                )
            )
        else:
            label_files.append(
                os.path.join(
                    os.sep.join(file_path.split("/")[:-2]),
                    "labels",
                    os.path.basename(file_path)[:-3] + "txt",
                )
            )
    data = {"class": [], "area": [], "image": [], "image_area": [], "Source": []}
    for i, file_path in enumerate(label_files):
        filename = os.path.basename(file_path)
        if filename.endswith(".json"):
            json_data = None
            with open(file_path) as fp:
                json_data = json.load(fp)

            for shape in json_data["shapes"]:
                if shape["shape_type"] != "rectangle":
                    continue
                norm_points = shape["points"]

                norm_points[0][0] = clamp(norm_points[0][0], 0, json_data["imageWidth"])
                norm_points[0][1] = clamp(
                    norm_points[0][1], 0, json_data["imageHeight"]
                )
                norm_points[1][0] = clamp(norm_points[1][0], 0, json_data["imageWidth"])
                norm_points[1][1] = clamp(
                    norm_points[1][1], 0, json_data["imageHeight"]
                )

                if shape["label"] not in labels_counters.keys():
                    if labels_path is None:
                        labels_counters[shape["label"]] = 0
                        labels_area_counters[shape["label"]] = 0
                    elif labels_path is not None:
                        continue
                data["class"].append(shape["label"])
                data["image"].append(filename)
                data["image_area"].append(
                    json_data["imageWidth"] * json_data["imageHeight"]
                )
                data["area"].append(
                    abs(norm_points[0][0] - norm_points[1][0])
                    * abs(norm_points[0][1] - norm_points[1][1])
                )
                if "schreder" in file_path:
                    data["Source"].append("Schreder")
                else:
                    data["Source"].append("Other")

        elif filename.endswith(".txt"):
            lines = []
            try:
                with open(os.path.join(os.path.dirname(dataset_txt), file_path)) as fp:
                    lines = fp.read().split("\n")
            except:
                continue
            im = cv2.imread(os.path.join(os.path.dirname(dataset_txt), image_files[i]))
            if im is None:
                continue

            image_height = im.shape[0]
            image_width = im.shape[1]

            for line in lines:
                values = line.split(" ")
                if len(values) == 5:
                    cls_id = int(values[0])
                    # other labels should be considered as "other" class
                    if cls_id >= len(id_to_label_dict.keys()):
                        c = "other"
                    else:
                        c = id_to_label_dict[cls_id]

                    w = float(values[3]) * image_width
                    h = float(values[4]) * image_height

                    data["class"].append(c)
                    data["image"].append(filename)
                    data["image_area"].append(image_width * image_height)
                    data["area"].append(w * h)
                    if "schreder" in file_path:
                        data["Source"].append("Schreder")
                    else:
                        data["Source"].append("Other")

    return pd.DataFrame(data)
