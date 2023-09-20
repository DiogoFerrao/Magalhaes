import os

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from yolov7.utils.datasets import LoadImages


def get_test_stats_df(eval_dir: str):
    """Get the test stats as a pandas dataframe.

    Args:
        eval_dir (str): Path to the directory where the test run was save.

    Returns:
        pd.DataFrame: Dataframe with the test stats.
    """
    # Load csv
    df = pd.read_csv(os.path.join(eval_dir, "stats.csv"))

    # Compute precision, recall and f1
    df["precision"] = df["correct"].divide(df["predicted"]).fillna(0)
    df["recall"] = df["correct"].divide(df["targets"]).fillna(0)
    df["f1"] = 2 * df["precision"].multiply(df["recall"]).divide(
        df["precision"].add(df["recall"])
    ).fillna(0)

    # Compute fraction of correct predictions across the different confidence thresholds
    df["fraction_correct"] = (
        df["thresholds_correct"].divide(df["thresholds_total"]).fillna(0)
    )
    return df


def get_worst_performing_images(
    eval_dir: str,
    n_images: int,
    names: list[str],
    min_num_pred: int = 0,
    ignore_classes: list[str] = [],
):
    """Get the images where the model performs the worst.
    The test script must have been run with the --save-incorrect flag.

    Args:
        eval_dir (str): Path to the directory where the test run was save.
        n_images (int): Number of images to return.
        names (list[str]): List of class names.
        min_num_pred (int, optional): Minimum number of predictions to consider an image. Defaults to 0.
        ignore_classes (list[str], optional): List of class names to ignore. Defaults to [].

    Returns:
        list[str]: List of paths to the images sorted from worst to best.
    """
    ignore_idxs = [names.index(class_name) for class_name in ignore_classes]

    df = get_test_stats_df(eval_dir)

    # Sort by fraction of correct predictions
    df = df.sort_values(by=["f1", "fraction_correct", "path"]).reset_index()

    # Filter by min_num_pred
    df = df[df["predicted"] >= min_num_pred]

    # Obtain paths to images
    images_paths = df[df["path"].str.contains("schreder")]["path"].values.tolist()
    sa, sb = (
        os.sep + "images" + os.sep,
        os.sep + "labels" + os.sep,
    )  # /images/, /labels/ substrings
    images_labels = [
        x.replace(sa, sb, 1).replace(x.split(".")[-1], "txt") for x in images_paths
    ]

    # Remove images that contain ignored classes
    filtered_images_paths = []
    for i in range(len(images_paths)):
        found_ignored_class = False
        for line in open(images_labels[i]).readlines():
            idx = line.split(" ")[0]
            if int(idx) in ignore_idxs:
                found_ignored_class = True
        if not found_ignored_class:
            filtered_images_paths.append(images_paths[i])

    return filtered_images_paths[:n_images]


def compute_imgs_representations(
    img_paths: list[str], model_backbone: nn.Module, device: str = "cuda"
):
    """Compute the representations of a list of images using a model backbone.

    Args:
        img_paths (list[str]): List of paths to the images.
        model_backbone (nn.Module): Model backbone.
        device (str, optional): Device to use. Defaults to "cuda".

    Returns:
        dict: Dictionary with the path of the image as key and the representation as value.
    """

    # Initialize Device
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    model_backbone.to(torch_device).eval()

    # Set Dataloader
    dataset = LoadImages(img_paths, img_size=640, stride=640)

    # Run inference
    img = torch.zeros((1, 3, 640, 640), device=torch_device)  # init img
    _ = model_backbone(img) if torch_device.type != "cpu" else None  # run once

    representations = {}

    for _, (path, img, _, _) in tqdm(enumerate(dataset)):
        img = torch.from_numpy(img).to(torch_device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            # Inference
            representations[path] = model_backbone(img).cpu()

    return representations


def compute_top_k_similar_reprs(
    img_repr: torch.Tensor,
    target_imgs_repr_dict: dict,
    k: int = 5,
    device: str = "cuda",
):
    """Compute the top k most similar representations using cosine similarity.

    Args:
        img_repr (torch.Tensor): Image representation.
        target_imgs_repr_dict (dict): Dictionary with the path of the image as key and the representation as value.

    Returns:
        list[str]: List of paths to the images.
    """
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")

    sim_dict = {}
    for path, curr_repr in target_imgs_repr_dict.items():
        sim_dict[path] = (
            F.cosine_similarity(img_repr.to(torch_device), curr_repr.to(torch_device))
            .cpu()
            .item()
        )
    return list(
        dict(sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)).keys()
    )[:k]


def compute_top_k_similar_imgs(
    img_paths: list[str], target_imgs: list[str], model_backbone: nn.Module, k: int = 5
):
    """Compute the top k most similar target images for each image in img_paths using cosine similarity.

    Args:
        img_paths (list[str]): List of paths to the images.
        target_imgs (list[str]): List of paths to the images we want to compare against.
        model_backbone (nn.Module): Model backbone.
        k (int, optional): Number of similar images to return. Defaults to 5.

    Returns:
        dict: Dictionary with the path of the image as key and a list of paths to the top k similar images as value.
    """
    imgs_repr_dict = compute_imgs_representations(img_paths, model_backbone)
    target_imgs_repr_dict = compute_imgs_representations(target_imgs, model_backbone)
    top_k_similar_imgs = {}
    for path, representation in tqdm(imgs_repr_dict.items()):
        top_k_similar_imgs[path] = compute_top_k_similar_reprs(
            representation, target_imgs_repr_dict, k
        )

    return top_k_similar_imgs
