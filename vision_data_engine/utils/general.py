import yaml


def get_class_names(filename):
    """Retrieves the class names from a file.

    Example:
        File coco.names:
        person
        bicycle
        car
        ...
        >>> get_class_names("coco.names")
        ['person', 'bicycle', 'car', ...]

    Args:
        filename (str): Path to the file containing the class names.

    Returns:
        names (list): List of class names.
    """
    # yaml extension
    if filename.endswith(".yaml") or filename.endswith(".yml"):
        with open(filename, "r") as f:
            names = yaml.load(f, Loader=yaml.FullLoader)["names"]
        return names
    elif filename.endswith(".txt") or filename.endswith(".names"):  # txt
        names = []
        for line in open(filename).readlines():
            names.append(line.strip())
        return names
    else:
        raise ValueError("File extension not supported.")
