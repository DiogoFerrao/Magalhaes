from typing import Union
import math
import random

from IPython.display import Image, display
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch

from yolov7.utils.general import xywh2xyxy

from vision_data_engine.utils.search import load_images_and_labels


def _get_ax(axs, row, col) -> plt.Axes:
    if isinstance(axs, np.ndarray):
        if isinstance(axs[0], np.ndarray):
            return axs[row, col]
        return axs[max(row, col)]
    return axs


def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in plt.rcParams["axes.prop_cycle"].by_key()["color"]]


def create_imgs_with_labels(img_paths: list[str], names: list[str]):
    """Create images with labels drawn on them.

    Args:
        img_paths (list[str]): List of paths to images.
        names (list[str]): List of class names.

    Returns:
        imgs (dict): Dictionary of images with labels.
    """
    images, labels = load_images_and_labels(img_paths)
    imgs = {}
    for img, targets, paths in zip(images, labels, img_paths):
        imgs[paths.split("/")[-1]] = draw_labels_to_image(img, targets, names=names)
    return imgs


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """Plots one bounding box on image img

    Args:
        x (list): [x1, y1, x2, y2] bounding box coordinates
        img (np.ndarray): image to plot on
        color (list, optional): color of the bounding box. Defaults to None.
        label (str, optional): label to put on the bounding box. Defaults to None.
        line_thickness (int, optional): thickness of the bounding box. Defaults to None.

    Returns:
        img (np.ndarray): image with bounding box
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl * 0.75, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl * 0.75,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def draw_labels_to_image(
    image: Union[np.ndarray, torch.Tensor],
    image_targets: Union[np.ndarray, torch.Tensor],
    names=None,
    max_size=640,
) -> np.ndarray:
    """Draw labels to an image

    Args:
        image (Union[np.ndarray, torch.Tensor]): uint8 RGB image of shape (h, w, c) to draw labels on
        image_targets (Union[np.ndarray, torch.Tensor]): targets for the image (IMG_ID, CLASS, X, Y, W, H, [CONF])
        names (list, optional): list of names for the labels. Defaults to None.
        max_size (int, optional): max size of the image. Defaults to 640.

    Returns:
        image (np.ndarray): image with labels
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().float().numpy()
    if isinstance(image_targets, torch.Tensor):
        image_targets = image_targets.cpu().numpy()

    # un-normalise
    if np.max(image) <= 1:
        image *= 255
    tl = 2  # line thickness
    h, w, _ = image.shape

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    colors = color_list()  # list of colors

    # change channel dimension
    if scale_factor < 1:
        image = cv2.resize(image, (w, h))

    if len(image_targets) > 0:
        conf = None
        # check for confidence presence (label vs pred)
        boxes = xywh2xyxy(image_targets[:, 2:6]).T
        conf = image_targets[:, 6] if image_targets.shape[1] == 7 else None
        classes = image_targets[:, 1].astype("int")
        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
        for j, box in enumerate(boxes.T):
            cls = int(classes[j])
            color = colors[cls % len(colors)]
            cls = names[cls] if names else cls
            label = "%s" % cls if conf is None else "%s %.1f" % (cls, conf[j])
            plot_one_box(box, image, label=label, color=color, line_thickness=tl)

    return image


def plot_images_with_labels(img_paths: list[str], names: list[str], n_columns=5):
    """Plot images with labels.

    Args:
        img_paths (list): List of images paths.
        names (list): List of class names.
        n_columns (int, optional): Number of columns. Defaults to 5.
    """
    n_rows = math.ceil(len(img_paths) / n_columns)
    missing = n_columns * n_rows
    fig, axs = plt.subplots(n_rows, n_columns, figsize=(6 * n_columns, 3.75 * n_rows))

    images, targets = load_images_and_labels(img_paths)
    for i, (img, targets, paths) in enumerate(zip(images, targets, img_paths)):
        row = i // n_columns
        column = i % n_columns
        img_name = paths.split("/")[-1]
        ax = _get_ax(axs, row, column)
        ax.imshow(draw_labels_to_image(img, targets, names=names))
        ax.set_title(img_name)
        ax.axis("off")
        missing -= 1

    for i in range(n_columns * n_rows - 1, n_columns * n_rows - 1 - missing, -1):
        row = i // n_columns
        column = i % n_columns
        ax = _get_ax(axs, row, column)
        axs[row][column].remove()

    return fig


def notebook_display_image(path):
    display(Image(mpimg.imread(path)))
