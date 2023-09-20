import os
import math
from typing import Union

import numpy as np
import pandas as pd
import cv2

from yolov7.utils.general import segments2boxes
from vision_data_engine.utils.general import get_class_names


def get_schreder_unlabeled_images(label_dirs: Union[list[str], str], dataset_dir: str):
    """Get images that are in dataset_dir but that are not in the dataset_path file

    Args:
        dataset_paths (list): List of paths to directories with labels
        dataset_dir (str): Path to the directory containing the images

    Returns:
        missing_images (list): List of paths to the missing images
    """
    existing_images = []
    missing_images = []

    if isinstance(label_dirs, str):
        label_dirs = [label_dirs]

    for label_dir in label_dirs:
        for label_path in os.listdir(label_dir):
            label_path = os.path.join(label_dir, label_path)
            img_path = label_path.replace("/labels/", "/images/", 1).replace(
                ".txt", ".jpg"
            )
            existing_images.append(img_path)

    for img_path in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, img_path)
        if img_path not in existing_images:
            missing_images.append(img_path)

    return missing_images


def get_schreder_unused_images(
    image_banks: Union[list[str], str], train_dataset: str, test_dataset: str
):
    """Get images that are in dataset_dir but that are not in the training and testing sets

    Args:
        image_banks (list): List of paths to directories with images
        train_dataset (str): Path to the .txt file containing the training images
        test_dataset (str): Path to the .txt file containing the testing images

    Returns:
        missing_images (list): List of paths to the missing images
    """
    existing_images = []
    missing_images = []

    if isinstance(image_banks, str):
        image_banks = [image_banks]

    while len(image_banks) > 0:
        image_bank = image_banks.pop()
        for image_path in os.listdir(image_bank):
            image_path = os.path.join(image_bank, image_path)
            if os.path.isdir(image_path):
                image_banks.append(image_path)
            existing_images.append(image_path)

    train_test_images = []
    for dataset in [train_dataset, test_dataset]:
        for image_path in open(dataset).readlines():
            image_path = image_path.strip()
            train_test_images.append(image_path)

    for image_path in existing_images:
        if image_path not in train_test_images:
            missing_images.append(image_path)

    return missing_images


def find_images_with_label(dataset_df, label):
    """Find images that have a specific label and return their paths

    Args:
        dataset_df (pd.DataFrame): DataFrame containing the dataset
        label (str): Label to search for

    Returns:
        images_with_label (list): List of paths to the images with the label
    """
    return dataset_df[dataset_df[label] == 1]["path"].values.tolist()


def load_detections_df(detection_banks: Union[list[str], str], class_names_path: str):
    """Load detections from a list of detection banks

    Args:
        detection_banks (list): List of paths to directories with detections
        class_names_path (str): Path to the class names file

    Returns:
        detections_df (pd.DataFrame): DataFrame containing the detections
    """
    names = get_class_names(class_names_path)

    data = {"path": []}
    for name in names:
        data[name] = []

    for detection_bank in detection_banks:
        for detection_path in os.listdir(detection_bank):
            detection_path = os.path.join(detection_bank, detection_path)
            data["path"].append(detection_path)

            for name in names:
                data[name].append(0)

            try:
                for line in open(detection_path).readlines():
                    cls_id = line.split(" ")[0]
                    data[names[int(cls_id)]][-1] += 1
            except:
                continue

    return pd.DataFrame(data)


def load_dataset_df(dataset_path: str, class_names_path: str):
    """Load a dataset file into a pandas DataFrame

    Args:
        dataset_path (str): Path to the dataset file
        class_names_path (str): Path to the class names file

    Returns:
        dataset_df (pd.DataFrame): DataFrame containing the dataset
    """
    names = get_class_names(class_names_path)

    data = {"path": []}
    for name in names:
        data[name] = []

    for img_path in open(dataset_path).readlines():
        label_path = (
            ".".join(img_path.replace("/images/", "/labels/").split(".")[:-1]) + ".txt"
        )
        data["path"].append(label_path)

        for name in names:
            data[name].append(0)

        try:
            for line in open(label_path).readlines():
                cls_id = line.split(" ")[0]
                data[names[int(cls_id)]][-1] += 1
        except:
            continue

    return pd.DataFrame(data)


def load_images_and_labels(paths: list[str], img_size: int = 640):
    """Load images and labels from a list of paths

    Args:
        paths (list): List of paths to the images

    Returns:
        images (list[np.ndarray]): List of images
        targets (list[np.ndarray]): List of targets
    """
    images = []
    labels = []
    for img_path in paths:
        label_path = (
            ".".join(img_path.replace("/images/", "/labels/").split(".")[:-1]) + ".txt"
        )
        images.append(load_image(img_path, img_size))
        label = load_labels(label_path)
        # When training the collater adds the index of the image to the label
        # We don't need it but we need to add a dummy value to the label
        label = np.concatenate([np.zeros((len(label), 1)), label], axis=1)
        labels.append(label)
    return images, labels


def load_image(path, img_size: int = 640, square: bool = False):
    """Loads an image from a path and resizes it to img_size

    Args:
        path (str): Path to the image
        img_size (int): Size to resize the image to

    Returns:
        img (np.ndarray): Image as a numpy array in RGB uint8 format with (h, w, c) shape
    """
    img = cv2.imread(path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    assert img is not None, "Image Not Found " + path
    h0, w0 = img.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

    if square:
        h_delta = img_size - img.shape[0]
        w_delta = img_size - img.shape[1]
        img = np.pad(
            img,
            (
                (math.floor(h_delta / 2), math.ceil(h_delta / 2)),
                (math.floor(w_delta / 2), math.ceil(w_delta / 2)),
                (0, 0),
            ),
            "constant",
        )
    return img


def load_labels(path):
    """Loads the labels from a label file

    Args:
        path (str): Path to the label file

    Returns:
        labels (np.ndarray): Labels as a numpy array
    """
    if not os.path.exists(path):
        return np.zeros((1, 6), dtype=np.float32)
    else:
        with open(path) as f:
            labels = [x.split() for x in f.read().strip().splitlines()]
            if any([len(x) > 8 for x in labels]):  # is segment
                classes = np.array([x[0] for x in labels], dtype=np.float32)
                segments = [
                    np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in labels
                ]  # (cls, xy1...)
                labels = np.concatenate(
                    (classes.reshape(-1, 1), segments2boxes(segments)), 1
                )  # (cls, xywh)
            labels = np.array(labels, dtype=np.float32)
    return labels


def filter_schreder_dataset_df(dataset_df):
    """Filter a dataset DataFrame to only keep the Schreder images

    Args:
        dataset_df (pd.DataFrame): DataFrame containing the dataset

    Returns:
        dataset_df (pd.DataFrame): DataFrame containing the filtered dataset
    """
    dataset_df = dataset_df[dataset_df["path"].str.contains("schreder")]
    return dataset_df
