import os
from datetime import datetime, timedelta


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
    names = []
    for line in open(filename).readlines():
        names.append(line.strip())
    return names


def find_audio_associated_image_files(
    audio_filename: str, image_filenames: list[str], time_offset: float = 0.0
):
    """Find the image files which timestamp is contained in the
        interval of the audio timestamp - 10 seconds and audio timestamp.

    Args:
        audio_filename (str): The filename to find the associated files.
        image_filenames (list[str]): The list of filenames to search for the associated files.
        time_offset (float, optional): The time offset in seconds between video and audio. Defaults to 0.0.

    Returns:
        list[str]: The list of associated files.
    """
    file_time = datetime.strptime(
        audio_filename.replace("_", ":").replace(".wav", ""), "%Y-%m-%d %H:%M:%S.%f%z"
    ) + timedelta(seconds=time_offset)
    res = []
    for curr_filename in image_filenames:
        curr_basename = os.path.basename(curr_filename)
        epsilon = timedelta(seconds=0.5)
        time_str = curr_basename.replace("_", ":").replace(".jpg", "")
        curr_file_time = datetime.strptime(time_str, "%Y-%m-%d::%H:%M:%S.%f%z")
        if (
            file_time - epsilon - timedelta(seconds=10)
            < curr_file_time
            < file_time + epsilon
        ):
            res.append(curr_filename)
    return res
