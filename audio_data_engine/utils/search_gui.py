import os

import pandas as pd

from IPython.display import display
import ipywidgets as widgets


class SearchGUI:
    SEARCHING_TEXT = "<h3><b>Searching...</b></h3>"
    WORKING_TEXT = "<h3><b>Ready</b></h3>"
    reload_options = False
    audios = {}
    dataset_df = None

    def start(self):
        elem = self.construct_gui()
        self.setup_callbacks()
        display(elem)

    def construct_gui(self):
        self.grid = widgets.GridspecLayout(8, 12, width="1500px", height="1080px")

        # Dataset display
        pd.set_option("display.max_colwidth", 1000)
        pd.set_option("display.max_rows", 10000)
        self.dataset_df_output = widgets.Output()
        self.dataset_header = widgets.HTML(
            value="<h3 style='margin-bottom: 6px'><b>Dataset</b></h3>",
        )

        # Loading test
        self.loading_text = widgets.HTML(
            value=self.WORKING_TEXT,
        )

        # Header
        self.dataset_path_input = widgets.Text(
            value="/media/magalhaes/sound/datasets/sound_1677779968.csv",
            placeholder="path",
        )
        dataset_path_input_container = widgets.VBox(
            [
                widgets.Label("Path to dataset .csv:"),
                self.dataset_path_input,
            ]
        )

        self.class_name_path_input = widgets.Text(
            value="car",
            placeholder="path",
        )
        class_name_path_input_container = widgets.VBox(
            [
                widgets.Label("Class name:"),
                self.class_name_path_input,
            ]
        )
        self.search_button = widgets.Button(
            description="Search",
            button_style="",
            icon="search",
        )
        self.header = widgets.HBox(
            [
                dataset_path_input_container,
                class_name_path_input_container,
                self.search_button,
            ]
        )

        # Sidebar
        self.audio_selection = widgets.Select(
            options=[],
            value=None,
            rows=40,
        )
        filter_header = widgets.Label("Select the datasets to filter:")
        self.dataset_filter = self.labels_checkboxes = widgets.VBox([])
        self.filter_btn = widgets.Button(
            description="Filter",
            button_style="",
            icon="filter",
        )
        selection_header = widgets.Label("Select audio file to listen:")
        audio_selection_container = widgets.VBox(
            [
                filter_header,
                self.dataset_filter,
                self.filter_btn,
                selection_header,
                self.audio_selection,
            ]
        )

        # Audio
        self.audio_player = widgets.Audio()
        self.audio_player_label = widgets.Label("Currently Playing:")
        audio_player_container = widgets.VBox(
            [self.audio_player_label, self.audio_player]
        )
        self.grid[:, :3] = audio_selection_container
        self.grid[0, 3:] = audio_player_container
        self.grid[1:, 3:] = widgets.VBox([self.dataset_header, self.dataset_df_output])

        return widgets.VBox([self.loading_text, self.header, self.grid])

    def setup_callbacks(self):
        self.search_button.on_click(self.search)
        self.audio_selection.observe(self.on_audio_selection_change)
        self.filter_btn.on_click(self.on_dataset_filter_change)

    def update_audios(self, dataset_df):
        self.audios = dict(
            zip(
                list(map(os.path.basename, dataset_df["audio_filename"].tolist())),
                dataset_df["audio_filename"],
            )
        )
        self.audio_selection.options = self.audios.keys()
        self.audio_selection.value = self.audio_selection.options[0]

    def search(self, _):
        dataset_path = self.dataset_path_input.value
        class_name = self.class_name_path_input.value

        if not os.path.exists(dataset_path):
            self.loading_text.value = "<h3><b>Dataset path does not exist</b></h3>"
            return

        if class_name == "":
            self.loading_text.value = "<h3><b>Class name is empty</b></h3>"
            return

        self.loading_text.value = self.SEARCHING_TEXT

        # Load dataset dataframe
        self.dataset_df = pd.read_csv(dataset_path)
        self.dataset_df = self.dataset_df[self.dataset_df[class_name] == 1]
        self.dataset_df = self.dataset_df.reset_index(drop=True)
        self.dataset_df_output.clear_output()
        with self.dataset_df_output:
            display(self.dataset_df)

        # Load filters
        self.dataset_filter.children = [
            widgets.Checkbox(
                value=True,
                description=label,
                disabled=False,
                indent=False,
            )
            for label in self.dataset_df["dataset"].unique().tolist()
        ]

        # Load audio
        self.update_audios(self.dataset_df)

        self.loading_text.value = self.WORKING_TEXT

    def on_audio_selection_change(self, obj):
        if obj["name"] == "value":
            self.audio_player_label.value = "Currently Playing: " + obj["new"]
            self.audio_player.set_value_from_file(self.audios[obj["new"]])

    def on_dataset_filter_change(self, obj):
        if self.dataset_df is not None:
            keep_datasets = [
                dataset.description
                for dataset in self.dataset_filter.children
                if dataset.value
            ]
            dataset_df = self.dataset_df[self.dataset_df["dataset"].isin(keep_datasets)]
            dataset_df = dataset_df.reset_index(drop=True)
            print(keep_datasets)
            self.update_audios(dataset_df)
            self.dataset_df_output.clear_output()
            with self.dataset_df_output:
                display(dataset_df)
