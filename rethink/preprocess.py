"""
1- load a file
2- pad the signal
3- extract the log spectrogram
4- normalize the spectrogram
5- save the spectrogram
"""
import argparse
import json
import math
import os
import pickle as pkl
from copy import deepcopy
from typing import List, Optional
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as TA_T
import torchvision
import torchvision.transforms as T
from audiomentations import (
    Compose,
    PitchShift as PitchShiftAug,
    AddGaussianNoise,
    AirAbsorption,
    Gain,
    LowPassFilter,
    ClippingDistortion,
    AddBackgroundNoise,
    TimeStretch
)
from tqdm import tqdm

from rethink.stft.effi import MelSTFT
from rethink.stft.torchlibrosa import LibrosaMelSpectrogram


class Loader:
    """Loader is responsible for loading an audio file"""

    def __init__(self, sample_rate, mono=True, device="cpu"):
        self.sample_rate = sample_rate
        self.mono = mono
        self.device = device

    def __call__(self, path):
        audio_time_series, source_sr = torchaudio.load(path)
        audio_time_series = torch.mean(audio_time_series, dim=0)
        audio_time_series = torchaudio.functional.resample(
            audio_time_series, orig_freq=source_sr, new_freq=self.sample_rate
        )

        return audio_time_series.to(self.device)


class Trimmer:
    def __init__(self, sample_rate: int, duration: int):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = self.sample_rate * self.duration

    def __call__(self, audio_time_series):
        # Keep 10 seconds from the middle
        if self.duration and len(audio_time_series) > self.n_samples:
            n_excess_samples = len(audio_time_series) - self.n_samples
            audio_time_series = audio_time_series[
                                math.floor(n_excess_samples / 2): len(audio_time_series)
                                                                  - math.ceil(n_excess_samples / 2)
                                ]
        return audio_time_series


class Padder:
    """Padder is responsible for padding the signal"""

    def __init__(self, sample_rate: int, duration: Optional[int] = None):
        self.duration = duration
        self.sample_rate = sample_rate

    def __call__(self, signal):
        if self.duration is not None:
            num_missing_items = int(self.sample_rate * self.duration) - len(signal)
            if num_missing_items > 0:
                return F.pad(
                    signal,
                    (
                        math.ceil(num_missing_items / 2),
                        math.floor(num_missing_items / 2),
                    ),
                )
        return signal


class LogMelSpectrogramExtractorModel(nn.Module):
    """LogMelSpectrogramExtractorModel is responsible for extracting the log spectrogram

    Args:
        sample_rate (int): sample rate of the audio
        n_mels (int): number of mel filterbanks
        length (int): width of the spectrogram
        duration (int): duration of the audio
        hop_sizes (list): hop sizes for the spectrogram
        window_sizes (list): window sizes for the spectrogram
        export (bool): if True, the extractor will use the mel spectrogram from the
        torchlibrosa library, which can be exported to ONNX. Otherwise, the extractor
        will use the mel spectrogram from torchaudio.
    """

    def __init__(
            self,
            sample_rate: int,
            n_mels: int,
            length: int,
            duration: int,
            hop_sizes=[10, 25, 50],
            window_sizes=[25, 50, 100],
            export=False,
    ):
        super().__init__()
        self.num_channels = 3
        self.sample_rate = sample_rate
        self.hop_sizes = hop_sizes
        self.window_sizes = window_sizes
        self.n_mels = n_mels
        self.length = length
        self.duration = duration
        self.export = export

        assert self.num_channels == len(self.hop_sizes) and self.num_channels == len(
            self.window_sizes
        )

        self.resize_transform = torchvision.transforms.Resize(
            (self.n_mels, self.length), antialias=None
        )

        n_fft = int(self.sample_rate / self.duration)
        self.mel_spec_transforms = []
        for i in range(3):
            window_length = int(round(self.window_sizes[i] * self.sample_rate / 1000))
            hop_length = int(round(self.hop_sizes[i] * self.sample_rate / 1000))
            if self.export:
                # mel_spec = ConvertibleSpectrogram(
                #     sr=self.sample_rate,
                #     n_fft=n_fft,
                #     hop_size=hop_length,
                #     win_size=window_length,
                #     n_mel=self.n_mels,
                # )
                # mel_spec.set_mode("DFT", "store")
                mel_spec = LibrosaMelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=window_length,
                    n_mels=self.n_mels,
                )
            else:
                mel_spec = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=n_fft,
                    win_length=window_length,
                    hop_length=hop_length,
                    n_mels=self.n_mels,
                )
            self.mel_spec_transforms.append(mel_spec)
        self.mel_spec_transforms = nn.ModuleList(self.mel_spec_transforms)

    def forward(self, signal: torch.Tensor):
        no_batch = signal.dim() == 1
        if no_batch:
            signal = signal.unsqueeze(0)

        N = signal.size(0)
        spectgrms = torch.zeros(
            (self.num_channels, N, self.n_mels, self.length), device=signal.device
        )
        for i in range(self.num_channels):
            # Spectogram
            spectgrm = self.mel_spec_transforms[i](signal)
            # Log spectogram
            eps = 1e-6
            log_spec = torch.log(spectgrm + eps)
            if self.export:
                log_spec = log_spec.unsqueeze(0)
                log_spec = self.resize_transform(log_spec)[0]
            else:
                log_spec = self.resize_transform(log_spec)

            log_spec = (log_spec + 4.5) / 5.0  # fast normalization
            spectgrms[i] = log_spec

        spectgrms = spectgrms.permute(1, 0, 2, 3)
        if no_batch:
            spectgrms = spectgrms.squeeze(0)
        return spectgrms


# class PitchShift:
#     def __init__(self, sample_rate: int, device="cpu"):
#         self.sample_rate = sample_rate
#         self.device = device

#     def __call__(self, waveform):
#         pitch_shift = int(torch.randint(-2, 2 + 1, (1,)).item())
#         return (
#             torchaudio.functional.pitch_shift(
#                 waveform, sample_rate=self.sample_rate, n_steps=pitch_shift
#             )
#             .to(self.device)
#             .squeeze(0)
#         )


class Roll(object):
    def __init__(self, shift_dims: tuple, dims: tuple, min: int, max: int):
        assert min < max
        self.shift_dims = np.array(shift_dims)
        self.dims = dims
        self.min = min
        self.max = max

    def __call__(self, x: torch.Tensor):
        multiplier = int(torch.rand(1).item() * self.max) + self.min
        return x.roll(tuple(self.shift_dims * multiplier), self.dims)


class MinMaxNormalizer:
    """MinMaxNormalizer is responsible for normalizing the spectrogram"""

    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def __call__(self, spectgrms):
        for spectgrm in spectgrms:
            norm_spectrogram = (spectgrm - spectgrm.min()) / (
                    spectgrm.max() - spectgrm.min()
            )
            spectgrm = norm_spectrogram * (self.max - self.min) + self.min
        return spectgrms


# class AugmentationProbWrapper:
#     def __init__(self, augmentation_fn, prob):
#         self.augmentation_fn = augmentation_fn
#         self.prob = prob

#     def __call__(self, signal):
#         if np.random.rand() < self.prob:
#             return self.augmentation_fn(signal)
#         return signal

class SpecAugment(object):
    def __init__(self, p: float, time_mask_param: int, freq_mask_param: int):
        self.p = p
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
    
    def __call__(self, spectgrm):
        # Only apply spec augment with probability p
        if np.random.rand() > self.p:
            return spectgrm        
        spec_aug = T.Compose([
            TA_T.TimeMasking(self.time_mask_param),
            TA_T.FrequencyMasking(self.freq_mask_param)
        ])
        return spec_aug(spectgrm)

class Oversampler:
    def __init__(self, p: float):
        self.probability = p

    def __call__(self, waveform, sample_rate):
        if np.random.rand() < self.probability:
            return waveform
        return None

class BackgroundNoise:
    def __init__(self, p: float, min_snr_db: float, max_snr_db: float):
        bg_noise_arr = [
            "/media/magalhaes/sound/background_noise/2022-10-17 16_50_45.670123+00_00.wav",
            "/media/magalhaes/sound/background_noise/4-164206-A-10.wav",
            "/media/magalhaes/sound/background_noise/2-82367-A-10.wav",
            "/media/magalhaes/sound/background_noise/1-17367-A-10.wav",
            "/media/magalhaes/sound/background_noise/5-157204-A-16.wav",
            "/media/magalhaes/sound/background_noise/4-163606-A-16.wav",
            "/media/magalhaes/sound/background_noise/3-134699-C-16.wav",
            "/media/magalhaes/sound/background_noise/2-109371-C-16.wav",
            "/media/magalhaes/sound/background_noise/2-109374-A-16.wav",
            "/media/magalhaes/sound/background_noise/1-69760-A-16.wav",
            "/media/magalhaes/sound/background_noise/1-51035-A-16.wav",
            "/media/magalhaes/sound/background_noise/1-137296-A-16.wav",
            "/media/magalhaes/sound/background_noise/1-47709-A-16.wav",
        ]
        self.bg_noise = AddBackgroundNoise(
            bg_noise_arr, p=0.2, lru_cache_size=len(bg_noise_arr), min_snr_in_db=min_snr_db, max_snr_in_db=max_snr_db
        )


    def __call__(self, waveform, sample_rate):
        waveform = self.bg_noise(waveform, sample_rate)
        return waveform
    
class Augmenter:
    """Augmenter is responsible for applying data augmentation"""

    def __init__(self, augmentations_file: str, extractor, sample_rate: int):
        if not os.path.exists(augmentations_file):
            raise ValueError(f"Augmentation configuration file {augmentations_file} does not exist!")

        self.sample_rate = sample_rate
        self.extractor = extractor
        
        with open(augmentations_file, "r") as f:
            self.config = json.load(f)

        self.waveform_augs = self.parse_augmentation("waveform_augmentations", self.get_waveform_aug_definitions())
        self.spectrogram_augs = self.parse_augmentation("spectrogram_augmentations", self.get_spectrogram_aug_definitions())
        
    def get_waveform_aug_definitions(self):
        return {
            "PitchShift": PitchShiftAug,
            "AddGaussianNoise": AddGaussianNoise,
            "AirAbsorption": AirAbsorption,
            "Gain": Gain,
            "LowPassFilter": LowPassFilter,
            "ClippingDistortion": ClippingDistortion,
            "BackgroundNoise": BackgroundNoise,
            "TimeStretch": TimeStretch,
            "Oversampler": Oversampler
        }

    def get_spectrogram_aug_definitions(self):
        return {
            "SpecAugment": SpecAugment,
        }

    def parse_augmentation(self, aug_key: str, definitions: dict):
        augmentations = []
        for aug_config in self.config[aug_key]:
            aug_name = aug_config["name"]
            aug_params = aug_config["params"]

            if aug_name in definitions:
                augmentation_fn = definitions[aug_name](**aug_params)
                print(f"Applying augmentation {augmentation_fn} with params {aug_params}")
                augmentations.append(augmentation_fn)
            else:
                print(f"Unknown augmentation '{aug_name}' for key '{aug_key}'")
        
        if aug_key == "waveform_augmentations":
            return Compose(augmentations)
        else:
            return T.Compose(augmentations)

    def __call__(self, entry) -> Optional[List]:

        augs_per_signal = int(self.config["augs_per_signal"])

        new_entries = []

        for _ in range(augs_per_signal):
            new_entry = deepcopy(entry)
            signal = new_entry["waveform"]
            augmented = False
        
            augmented_signal = self.waveform_augs(signal, sample_rate=self.sample_rate)
            if not np.array_equal(signal, augmented_signal):
                augmented = True

            # Convert waveform to torch and apply transformations
            augmented_signal = torch.tensor(augmented_signal, device="cuda")
            spectgrms = self.extractor(augmented_signal)

            # Never actually a problem because spec augments dont take sample_rate as args
            augmented_spectgrms = self.spectrogram_augs(spectgrms)

            # Check if we had any augmentation effect
            if torch.equal(augmented_spectgrms, spectgrms) and not augmented:
                continue
            
            converted_spectgrms = augmented_spectgrms.cpu().numpy()

            new_entry["waveform"] = augmented_signal.cpu().numpy()
            new_entry["spectograms"] = converted_spectgrms
            new_entry["augmented"] = True

            new_entries.append(new_entry)
        return new_entries

class Packager:
    """Packager is responsible for packaging the information"""

    def __init__(self, save_waveform=False, save_spectogram=True):
        self.save_waveform = save_waveform
        self.save_spectogram = save_spectogram

    def package(self, filename, signal, spectgrms, labels, dataset):
        entry = {}

        entry["augmented"] = False
        entry["filename"] = filename
        entry["waveform"] = signal.cpu().numpy()
        entry["spectograms"] = np.array(spectgrms)
        entry["dataset"] = dataset
        entry["target"] = labels
        return entry

    def dump(self, output_path, output_name, data):
        list_of_dicts = data

        if not self.save_waveform:
            key_to_remove = "waveform"

            # Using list comprehensions to remove the specified key from all dictionaries
            list_of_dicts = [
                {k: v for k, v in dictionary.items() if k != key_to_remove}
                for dictionary in data
            ]

        with open(os.path.join(output_path, output_name), "wb") as handler:
            pkl.dump(list_of_dicts, handler, protocol=pkl.HIGHEST_PROTOCOL)


class Preprocessor:
    """Preprocessor is responsible for preprocessing the audio files"""

    def __init__(self, loader, trimmer, padder, extractor, packager):
        self.loader : Loader = loader
        self.trimmer : Trimmer = trimmer
        self.padder : Padder = padder
        self.extractor = extractor
        self.packager : Packager = packager

    def preprocess(self, path, labels, dataset):
        signal = self.loader(path)
        signal = self.trimmer(signal)
        signal = self.padder(signal)
        spectgrms = self.extractor(signal).cpu()
        return self.packager.package(path, signal, spectgrms, labels, dataset)


def preprocess(
        csv_file,
        output_dir="/media/magalhaes/sound/spectograms",
        sample_rate=22050,
        n_mels=128,
        image_length=256,
        duration=10.0,
        effi_extractor=False,
        export_extractor=False,
        waveform_only=False,
        labels_path="./data/schreder.names",
        device="cuda",
        augment=False,
        augmentations_file="./config/augmentations.json",
        class_imbalance_augment=False,
):
    audios_df = pd.read_csv(csv_file, skipinitialspace=True)

    dataset_name = os.path.basename(csv_file).split(".")[0]

    os.makedirs(output_dir, exist_ok=True)

    label_to_id = {}
    for i, label in enumerate(open(labels_path).readlines()):
        label_to_id[label.strip()] = i

    # Get number of splits and
    splits_arr = list(audios_df["split"].unique())
    audio_splits = [audios_df.loc[audios_df["split"] == i] for i in splits_arr]

    loader = Loader(sample_rate=sample_rate, device=device)
    trimmer = Trimmer(sample_rate, duration)
    padder = Padder(sample_rate=sample_rate, duration=duration)

    if effi_extractor:
        extractor = MelSTFT(n_mels=n_mels, sr=sample_rate).to(device)
    else:
        extractor = LogMelSpectrogramExtractorModel(
            sample_rate,
            n_mels,
            image_length,
            duration,
            export=export_extractor,
        ).to(device)

    packager = Packager(save_waveform=waveform_only, save_spectogram=not waveform_only)

    preprocessor = Preprocessor(loader, trimmer, padder, extractor, packager)
    
    augmenters = {}

    # List of all class columns
    class_columns = list(label_to_id.keys())

    # Calculate the sum for each class column
    class_counts = audios_df[class_columns].sum().to_dict()

    if class_imbalance_augment:
        for class_name, class_id in label_to_id.items():
            # strip augmentations_file extension
            filename = os.path.splitext(augmentations_file)[0]
            # append class name to filename
            filename += "_" + class_name
            # append extension
            filename += ".json"
            augmenters[class_id] = Augmenter(filename, extractor=extractor, sample_rate=sample_rate)
    else:
        augmenters["default"] = Augmenter(augmentations_file, extractor=extractor, sample_rate=sample_rate)

    for j, dataset_split in enumerate(audio_splits):
        values = []
        augmented_values = []
        split_filename = f"{dataset_name}_{splits_arr[j]}.pkl"
        print(f"Processing {split_filename}")
        with tqdm(total=len(dataset_split)) as pbar:
            for i, row in dataset_split.iterrows():
                # Get labels
                labels = [
                    int(row[key])
                    for key in audios_df.columns
                    if key in label_to_id.keys()
                ]

                # Preprocess data
                entry = preprocessor.preprocess(
                    row["audio_filename"], labels, row["dataset"]
                )
                

                values.append(entry)
                if augment:
                    target = entry["target"]
                    # 1. Extract classes present in the entry
                    classes_present_in_entry = [class_columns[i] for i, label in enumerate(target) if label == 1]

                    # 2. Get counts for these classes
                    class_counts_for_entry = {cls: class_counts[cls] for cls in classes_present_in_entry}

                    if len(class_counts_for_entry) == 0:
                        continue

                    # 3. Determine the least represented class
                    least_represented_class = min(class_counts_for_entry, key=class_counts_for_entry.get)

                    # 4. Get the label for the least represented class
                    least_represented_label = label_to_id[least_represented_class]

                    if class_imbalance_augment:
                        # get the least represented class_name in the entry
                        # get the augmenter for that label
                        augmenter = augmenters[least_represented_label]
                    else:
                        augmenter = augmenters["default"]


                    augmented_entries : Optional[List] = augmenter(entry)
                    if augmented_entries is not None:
                        augmented_values.extend(augmented_entries)
                    
                # Print memory usage of each variable
                # print(f"Entry: {sys.getsizeof(entry)}")
                # print(f"Values: {sys.getsizeof(values)}")
                # print(f"Augmented values: {sys.getsizeof(augmented_values)}")
                # print(f"Augmenters: {sys.getsizeof(augmenters)}")

                pbar.update(1)

        if augment:
            print(f"Augmenting data, {len(augmented_values)} entries")
            # Combine augmented and non augmented data
            values.extend(augmented_values)

            # Print the number of augmented entries per class
            for class_name, class_id in label_to_id.items():
                print(f"{class_name}: {sum([1 for entry in augmented_values if entry['target'][class_id] == 1])}")

            # Print original data size and augmented data size
            print(f"Original data size: {len(values) - len(augmented_values)}")
            print(f"Augmented data size: {len(augmented_values)}")
            print(f"Total data size: {len(values)}")

        print(f"Saved to {output_dir}")

        # Save data
        preprocessor.packager.dump(output_dir, split_filename, values)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files into 3 channel spectrograms and stores the result in an npy file"
    )
    parser.add_argument("csv_file", type=str, help="csv file with the dataset")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate ")
    parser.add_argument("--n_mels", type=int, default=128,
                        help="Number of mel filterbanks which defines the height of the spectrogram")
    parser.add_argument("--image_length", type=int, default=256,
                        help="Width of 3 channel spectrogram after preprocessing")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duration of audios (larger audios will clipped and smaller will be padded)")
    parser.add_argument("--effi_extractor", action="store_true", help="Use mel spectrogram extractor from Effi")
    parser.add_argument("--export_extractor", action="store_true",
                        help="Use mel spectrogram extractor from torchlibrosa that is compatible with ONNX")
    parser.add_argument("--waveform_only", action="store_true", help="Only save waveform")
    parser.add_argument("--labels_path", type=str, default="./data/schreder.names",
                        help="Path to text file with label names")
    parser.add_argument(
        "--output_dir", type=str, default="/media/magalhaes/sound/spectograms", help="Path to output directory"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--augment", action="store_true", help="Wether to augment data")
    parser.add_argument("--augmentations_file", type=str, default="./config/augmentations.json",
                        help="Path to JSON file with augmentations")

    args = parser.parse_args()
    # fmt: on

    preprocess(
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        image_length=args.image_length,
        duration=args.duration,
        effi_extractor=args.effi_extractor,
        export_extractor=args.export_extractor,
        waveform_only=args.waveform_only,
        labels_path=args.labels_path,
        device=args.device,
        augment=args.augment,
        augmentations_file=args.augmentations_file,
    )
