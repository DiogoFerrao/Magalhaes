import os
from typing import Optional

import pandas as pd

from IPython.display import display
import ipywidgets as widgets
from ipywidgets.widgets.widget_media import Audio

from audio_data_engine.utils.general import (
    find_audio_associated_image_files,
    get_class_names,
)


class Store:
    _images = []
    _dataset_df = None
    _audios = []

    def set_images(self, x):
        self._images = x

    def get_images(self):
        return self._images

    def set_dataset_df(self, x):
        self._dataset_df = x

    def get_dataset_df(self):
        return self._dataset_df

    def set_audios(self, x):
        self._audios = x

    def get_audios(self):
        return self._audios


class AnnotationGUI:
    LOADING_TEXT = "<h3><b>Loading...</b></h3>"
    WORKING_TEXT = "<h3><b>Ready</b></h3>"
    names = []
    vision_names = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "emergency_vehicle",
        "bus",
        "other",
        "truck",
    ]
    reload_options = False

    def start(self):
        self.store = Store()
        elem = self.construct_gui()
        self.setup_callbacks()
        display(elem)

    def construct_gui(self):
        self.grid = widgets.GridspecLayout(8, 12, width="1500px", height="1080px")

        # Dataset display
        pd.set_option("display.max_colwidth", 1000)
        self.dataset_df_output = widgets.Output()
        self.dataset_header = widgets.HTML(
            value="<h3 style='margin-bottom: 6px'><b>Dataset</b></h3>",
        )

        # HEADER
        self.audio_dir_input = widgets.Text(
            value="/media/magalhaes/schreder_sound/11_04_2023",
            placeholder="some/path",
        )
        audio_dir_input_container = widgets.VBox(
            [
                widgets.Label("Path to directory with audio files to annotate:"),
                self.audio_dir_input,
            ]
        )

        self.image_dir_input = widgets.Text(
            value="/media/magalhaes/schreder/images/2023_04_11/",
            placeholder="path",
        )
        image_dir_input_container = widgets.VBox(
            [
                widgets.Label("Path to directory with corresponding images:"),
                self.image_dir_input,
            ]
        )

        self.class_name_path_input = widgets.Text(
            value="../rethink/data/schreder.names",
            placeholder="path",
        )
        class_name_path_input_container = widgets.VBox(
            [
                widgets.Label("Path to class name file:"),
                self.class_name_path_input,
            ]
        )

        self.time_offset_input = widgets.Text(
            value="0",
            placeholder="Time offset",
        )
        time_offset_input_container = widgets.VBox(
            [
                widgets.Label("Time offset between video and audio:"),
                self.time_offset_input,
            ]
        )

        self.load_button = widgets.Button(
            description="Load",
            button_style="",
            icon="spinner",
        )

        self.save_button = widgets.Button(
            description="Save",
            button_style="",
            icon="save",
            disabled=True,
        )

        header = widgets.HBox(
            [
                audio_dir_input_container,
                image_dir_input_container,
                class_name_path_input_container,
                time_offset_input_container,
                self.load_button,
                self.save_button,
            ]
        )

        # SIDEBAR
        self.audio_selection = widgets.Select(
            options=[],
            value=None,
            rows=40,
        )
        selection_header = widgets.Label("Select audio file to annotate:")
        audio_selection_container = widgets.VBox(
            [selection_header, self.audio_selection]
        )

        # Audio
        self.audio_player = Audio()
        self.audio_player_label = widgets.Label("Currently Playing:")
        audio_player_container = widgets.VBox(
            [self.audio_player_label, self.audio_player]
        )

        # Labels from image detection
        self.image_labels = widgets.Label()
        image_labels_container = widgets.VBox(
            [widgets.Label("YOLO predictions:"), self.image_labels]
        )

        audio_and_labels_container = widgets.HBox(
            [audio_player_container, image_labels_container]
        )

        # Image grid
        image_grid = widgets.GridspecLayout(4, 3)
        self.image_grid_elems = []
        for i in range(12):
            self.image_grid_elems.append(widgets.Image())
            image_grid[i // 3, i % 3] = self.image_grid_elems[i]

        # Labeling
        labeling_header = widgets.Label("Select labels:")
        self.labels_checkboxes = widgets.VBox([])
        self.confirm_btn = widgets.Button(
            description="Confirm",
            button_style="",
            icon="check",
        )
        labeling_container = widgets.VBox(
            [labeling_header, self.labels_checkboxes, self.confirm_btn]
        )

        # Loading text
        self.loading_text = widgets.HTML(
            value=self.WORKING_TEXT,
        )

        self.grid[0, :] = header
        self.grid[1:, :3] = audio_selection_container
        self.grid[1, 3:10] = audio_and_labels_container
        self.grid[2:, 3:10] = image_grid
        self.grid[2:, 10:] = labeling_container

        return widgets.VBox(
            [self.loading_text, self.grid, self.dataset_header, self.dataset_df_output]
        )

    def setup_callbacks(self):
        self.audio_selection.observe(self.on_audio_selection)
        self.load_button.on_click(self.load_directories)
        self.confirm_btn.on_click(self.save_audio_labels)
        self.save_button.on_click(self.save_dataset_df)

    def get_audio_associated_images(self, audio_path):
        """Loads audio associated images to image grid.

        Args:
            audio_path (str): Path to audio file.

        Returns:
            list: List of image paths.
        """
        audio_filename = os.path.basename(audio_path)
        images = sorted(
            find_audio_associated_image_files(
                audio_filename,
                self.store.get_images(),
                int(self.time_offset_input.value),
            )
        )
        return images

    def get_audio_labels(self, audio_path):
        """Loads audio labels to checkboxes."""
        dataset_df = self.store.get_dataset_df()
        row = dataset_df.loc[dataset_df["audio_filename"] == audio_path]
        for label in self.labels_checkboxes.children:
            label.value = bool(
                row[label.description].values[0] if row.shape[0] != 0 else 0
            )

    def get_audio_full_path(self, audio_filename):
        """Returns the full path to the audio file."""
        return os.path.join(self.audio_dir_input.value, audio_filename)

    def get_current_audio_full_path(self):
        """Returns the full path to the current selected audio file."""
        return self.get_audio_full_path(
            self.get_select_option_name(self.audio_selection.value)
        )

    def get_dataset_path(self):
        """Returns the path to the dataset csv file."""
        return self.audio_dir_input.value.rstrip("/") + "_labels.csv"

    def save_audio_labels(self, _):
        """Callback for confirm button. Saves audio labels to dataset dataframe."""
        dataset_df = self.store.get_dataset_df()
        row = dataset_df.loc[
            dataset_df["audio_filename"] == self.get_current_audio_full_path()
        ]
        if row.shape[0] != 0:
            dataset_df.drop(row.index, inplace=True)
        new_row = {
            "audio_filename": self.get_current_audio_full_path(),
            "dataset": self.audio_dir_input.value.rstrip("/").split("/")[-1],
        }
        for label in self.labels_checkboxes.children:
            new_row[label.description] = int(label.value)
        dataset_df = pd.concat(
            [dataset_df, pd.DataFrame(new_row, index=[0])], ignore_index=True
        )
        # sort by audio filename
        dataset_df = dataset_df.sort_values(by=["audio_filename"], ignore_index=True)
        self.store.set_dataset_df(dataset_df)
        self.update_sound_select_options(
            self.get_select_option_name(self.audio_selection.value)
        )
        self.dataset_df_output.clear_output()
        with self.dataset_df_output:
            display(self.store.get_dataset_df())

    def save_dataset_df(self, _):
        """Saves dataset dataframe to csv."""
        dataset_df = self.store.get_dataset_df()
        dataset_df.to_csv(self.get_dataset_path(), index=False)

    def format_select_options(self, option: str, state: bool):
        """Returns a string with the format "color_chr option_name". Where color_chr
        is a unicode character that represents if the option is labeled or not.

        Args:
            option (str): Option string.
            state (bool): If the option is labeled or not.

        Returns:
            str: Formatted option string.
        """
        c_base = int("1F534", base=16)
        return f"{chr(c_base+state)} {option}"

    def get_select_option_name(self, option: Optional[str]) -> str:
        """Returns the name of the option.

        Option is a string with the format "color_chr option_name". Where color_chr
        is a unicode character that represents if the option is labeled or not.

        Args:
            option (Optional[str]): Option string.

        Returns:
            str: Option name.
        """
        if option is None:
            return ""
        return " ".join(option.split(" ")[1:])

    def update_sound_select_options(self, value: str):
        """Updates audio selection options.

        Args:
            value (str): New value to set the audio selection.
        """
        options = self.store.get_audios()
        dataset = self.store.get_dataset_df()
        state = []
        # check options that are already in dataset
        for i, option in enumerate(options):
            state.append(
                self.get_audio_full_path(option) in dataset["audio_filename"].values
            )
        colored_options = [
            self.format_select_options(o, s) for s, o in zip(state, options)
        ]
        self.audio_selection.options = colored_options
        self.audio_selection.value = self.format_select_options(
            value, self.get_audio_full_path(value) in dataset["audio_filename"].values
        )

    def get_images_labels(self, images: list[str]):
        """Returns the labels of an image.

        Args:
            images (list): List of image paths.

        Returns:
            list: List of image labels.
        """
        label_set = set()
        for image_path in images:
            detection_path = image_path.replace("/images/", "/detections/").replace(
                ".jpg", ".txt"
            )
            detection_path = os.path.join(
                os.path.dirname(detection_path),
                "labels",
                os.path.basename(detection_path),
            )
            df = pd.read_csv(detection_path, sep=" ", header=None)
            labels = df.iloc[:, 0].unique().tolist()
            label_set.update(labels)
        return list(label_set)

    def update_image_grid(self, images: list[str]):
        """Updates image grid.

        Args:
            images (list): List of image paths.
        """
        for i, image_path in enumerate(images):
            self.image_grid_elems[i].set_value_from_file(image_path)

    def update_image_labels_elem(self, labels: list[int]):
        """Updates image labels element.

        Args:
            labels (list): List of label ids.
        """
        self.image_labels.value = ", ".join(
            list(map(lambda x: self.vision_names[x], labels))
        )

    def on_audio_selection(self, obj: dict):
        """Callback for audio selection. Loads audio, labels and images.

        Args:
            obj (dict): Change object.
        """
        if obj["name"] == "_options_labels":
            if len(obj["old"]) > 0 and self.get_select_option_name(
                obj["old"][0]
            ) == self.get_select_option_name(obj["old"][0]):
                self.reload_options = True
        if obj["name"] == "value":
            if self.reload_options:
                self.reload_options = False
            elif (
                self.get_select_option_name(obj["old"])
                != self.get_select_option_name(obj["new"])
                and self.audio_selection.value is not None
            ):
                self.audio_player_label.value = "Loading..."
                self.loading_text.value = self.LOADING_TEXT
                audio_path = self.get_current_audio_full_path()
                self.audio_player.set_value_from_file(audio_path)
                images = self.get_audio_associated_images(audio_path)
                image_labels = self.get_images_labels(images)
                self.update_image_labels_elem(image_labels)
                self.get_audio_labels(audio_path)
                self.update_image_grid(images)
                self.audio_player_label.value = (
                    "Currently Playing: "
                    + self.get_select_option_name(self.audio_selection.value)
                )
                self.loading_text.value = self.WORKING_TEXT

    def load_labels(self, names: list[str]):
        """Loads labels to checkboxes.

        Args:
            names (list): List of label names.
        """
        self.labels_checkboxes.children = ()
        for name in names:
            self.labels_checkboxes.children = (
                *self.labels_checkboxes.children,
                widgets.Checkbox(description=name),
            )

    def load_directories(self, _):
        """Callback for load button. Loads audio files, image files and labels."""
        audio_files = sorted(
            [f for f in os.listdir(self.audio_dir_input.value) if f.endswith(".wav")]
        )
        if len(audio_files) > 0:
            self.loading_text.value = self.LOADING_TEXT
            self.save_button.disabled = False
            self.audio_player_label.value = "Loading..."

            self.names = get_class_names(self.class_name_path_input.value)
            self.store.set_audios(audio_files)

            # Load dataset dataframe
            df_path = self.get_dataset_path()
            if not os.path.exists(df_path):
                dataset_df = pd.DataFrame(
                    columns=["audio_filename", "dataset"] + self.names
                )
            else:
                dataset_df = pd.read_csv(df_path)
            self.store.set_dataset_df(dataset_df)
            with self.dataset_df_output:
                display(self.store.get_dataset_df())
            # Load labels checkboxes
            self.load_labels(self.names)
            # load list of image files
            self.store.set_images(
                [
                    os.path.join(self.image_dir_input.value, f)
                    for f in os.listdir(self.image_dir_input.value)
                    if f.endswith(".jpg")
                ]
            )
            self.update_sound_select_options(audio_files[0])
        else:
            self.audio_selection.options = ["No Files Found"]
            self.audio_selection.value = None
