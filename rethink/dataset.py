from typing import Optional, Callable, Union

import numpy as np
import pandas as pd
import pickle

import torch.utils.data as tdata
import torch
import torchaudio

from audiomentations import (
    AddBackgroundNoise,
    AddGaussianSNR,
)

from rethink.preprocess import (
    Trimmer,
    Padder,
    LogMelSpectrogramExtractorModel,
)


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


class WaveformAugmentations:
    def __init__(self):
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
            bg_noise_arr, p=0.2, lru_cache_size=len(bg_noise_arr)
        )
        self.gaussian_noise = AddGaussianSNR(
            min_snr_in_db=3.0, max_snr_in_db=30.0, p=0.2
        )

    def __call__(self, waveform):
        if np.random.rand() > 0.8:
            waveform = self.bg_noise(waveform, 22050)
        else:
            waveform = self.gaussian_noise(waveform, 22050)

        return waveform


class AugmentedMelSpectrogram(object):
    def __init__(self, bins=128, duration=10, sr=22050, length=250, device="cpu"):
        self.window_length = [25, 50, 100]
        self.hop_length = [10, 25, 50]
        self.fft = int(sr / 10)
        self.melbins = bins
        self.sr = sr
        self.length = length
        self.duration = duration
        self.device = device

    def __call__(self, audio):
        audio = audio.to(self.device)
        for module in self.modules:
            audio = module(audio)

        return audio.cpu()


class AudioDataset(tdata.Dataset):
    def __init__(
        self,
        pkl_dirs,
        image_size=(128, 250),
        waveform_transforms=None,
        spectrogram_transforms=None,
        filter_datasets=[],
    ):
        if isinstance(pkl_dirs, str):
            pkl_dirs = [pkl_dirs]
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.image_size = image_size

        self.extractor = LogMelSpectrogramExtractorModel(
            sample_rate=22050, n_mels=128, length=250, duration=10
        )

        self.data = []
        tmp_data = []
        for pkl_dir in pkl_dirs:
            with open(pkl_dir, "rb") as f:
                tmp_data.extend(pickle.load(f))
        if len(filter_datasets) != 0:
            for d in tmp_data:
                if d["dataset"] in filter_datasets:
                    self.data.append(d)
        else:
            self.data = tmp_data

        print("Applying spectrogram transforms")
        print(self.spectrogram_transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # 10 seconds, 22.05kHz
        # waveform = entry["waveform"]
        spectrogram = torch.Tensor(entry["spectograms"])
        # spectrogram = self.extractor(torch.Tensor(waveform))

        if self.spectrogram_transforms is not None:
            # augment audio and create new spectogram
            spectrogram = self.spectrogram_transforms(spectrogram)

        target = torch.tensor([entry["target"]], dtype=torch.float32)

        # if self.waveform_transforms is not None:
        #     if "2022" not in entry["dataset"] and "2023" not in entry["dataset"]:
        #         waveform = self.waveform_transforms(waveform)
        #     # student_waveform = self.waveform_transforms(waveform)
        # augment audio and create new spectogram
        # teacher_waveform = torch.Tensor(waveform)
        # teacher_spectrogram = self.extractor(teacher_waveform)
        # if self.spectrogram_transforms is not None:
        #     teacher_spectrogram = self.spectrogram_transforms(teacher_spectrogram)
        # return (torch.Tensor(student_waveform), teacher_waveform, target)
        # return (torch.Tensor(waveform), target)

        return (spectrogram, target)

class RawAudioDataset(tdata.Dataset):
    def __init__(
        self,
        dataset_df: pd.DataFrame,
        names: list[str],
        sample_rate: int,
        duration: int,
        extractor: LogMelSpectrogramExtractorModel,
        waveform_transforms: Optional[Callable],
        spectrogram_transforms: Optional[Callable],
        filter_datasets=[],
    ):
        self.dataset_df = dataset_df
        paths = []
        self.targets = torch.tensor([])
        self.waveforms = torch.tensor([])
        n_targets = len(names)
        n_audios = len(dataset_df)

        self.extractor = extractor
        self.waveform_transforms = waveform_transforms
        self.spec_transforms = spectrogram_transforms
        self.sample_rate = sample_rate
        #Print the transforms for debugging
        print("Waveform transforms:")
        print(self.waveform_transforms)

        print("Spectrogram transforms:")
        print(self.spec_transforms)

        self.filter_datasets = filter_datasets

        trimmer = Trimmer(sample_rate, duration)
        padder = Padder(sample_rate, duration)

        self.waveforms = torch.zeros((n_audios, sample_rate * duration))
        # self.targets = torch.zeros((n_audios, n_targets))
        self.targets = torch.tensor(dataset_df[names].values, dtype=torch.float32)
        # A spectrogram is shaped torch.Size([3, 128, 250])
        self.spectrograms = torch.zeros((n_audios, 3, 128, 250))

        print("Loading dataset to RAM")
        for i, row in enumerate(dataset_df.itertuples()):
            if i % 1000 == 0:
                print(i)
            path = row.audio_filename
            paths.append(path)

            waveform, source_sr = torchaudio.load(path)
            waveform = waveform.to("cuda")
            if waveform.shape[1] == 0:
                waveform = torch.zeros((1, sample_rate * duration))
            waveform = torch.mean(waveform, dim=0)
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=source_sr, new_freq=sample_rate
            )

            # (1,L) -> (L)
            waveform = waveform.squeeze(0)

            waveform = trimmer(waveform)
            waveform = padder(waveform)

            if len(self.filter_datasets) != 0:
                if row.dataset not in self.filter_datasets:
                    continue
            
            waveform = waveform.cpu()

            self.waveforms[i] = waveform

            if self.waveform_transforms is not None:
                waveform = self.waveform_transforms(samples= waveform.numpy(), sample_rate=self.sample_rate)
            
            waveform = torch.Tensor(waveform)

            spectrogram = self.extractor(waveform)

            if self.spec_transforms is not None:
                spectrogram = self.spec_transforms(spectrogram)

            self.spectrograms[i] = spectrogram.cpu()

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):

        spectrogram = self.spectrograms[idx]
        target = self.targets[idx]

        return (
            spectrogram,
            target,
        )


class DistilAudioDataset(tdata.Dataset):
    def __init__(
        self,
        dataset_csv: str,
        teacher_logits_path: Optional[str],
        names: list[str],
        sample_rate: float,
        duration: float,
        waveform_transforms: Optional[Callable],
    ):
        dataset_df = pd.read_csv(dataset_csv)
        paths = []
        self.targets = torch.tensor([])
        self.waveforms = torch.tensor([])
        n_targets = len(names)
        n_audios = len(dataset_df)

        self.waveform_transforms = waveform_transforms

        trimmer = Trimmer(sample_rate, duration)
        padder = Padder(sample_rate, duration)

        self.waveforms = torch.zeros((n_audios, sample_rate * duration))
        self.targets = torch.zeros((n_audios, n_targets))

        self.teacher_logits_targets = None
        if teacher_logits_path is not None:
            self.teacher_logits_targets = np.load(teacher_logits_path)

        print("Loading dataset to RAM")
        for i, row in enumerate(dataset_df.itertuples()):
            if i % 1000 == 0:
                print(i)
            path = row.filepath
            labels = row.label_id
            paths.append(path)

            waveform, source_sr = torchaudio.load(path)
            waveform = waveform.to("cuda")
            if waveform.shape[1] == 0:
                waveform = torch.zeros((1, sample_rate * duration))
            waveform = torch.mean(waveform, dim=0)
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=source_sr, new_freq=sample_rate
            )

            # (1,L) -> (L)
            waveform = waveform.squeeze(0)

            waveform = trimmer(waveform)
            waveform = padder(waveform)

            target = torch.zeros((n_targets))

            for j, name in enumerate(names):
                target[j] = 1 if name in labels else 0

            self.waveforms[i] = waveform.cpu()
            self.targets[i] = target

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]

        if self.waveform_transforms is not None:
            waveform = self.waveform_transforms(waveform)

        if self.teacher_logits_targets is None:
            return (
                waveform,
                self.targets[idx],
            )
        else:
            return (
                waveform,
                self.targets[idx],
                self.teacher_logits_targets[idx],
            )


class TeacherAudioDataset(tdata.Dataset):
    def __init__(self, dataset_csv, names, sample_rate, duration):
        dataset_df = pd.read_csv(dataset_csv)
        paths = []
        self.targets = torch.tensor([])
        self.waveforms = torch.tensor([])
        n_targets = len(names)
        n_audios = len(dataset_df)

        trimmer = Trimmer(sample_rate, duration)
        padder = Padder(sample_rate, duration)

        self.waveforms = torch.zeros((n_audios, sample_rate * duration))
        self.targets = torch.zeros((n_audios, n_targets))

        print("Loading dataset to RAM")
        for i, row in enumerate(dataset_df.itertuples()):
            if i % 1000 == 0:
                print(i)
            path = row.filepath
            paths.append(path)
            waveform, source_sr = torchaudio.load(path)
            waveform = waveform.to("cuda")
            waveform = torch.mean(waveform, dim=0)
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=source_sr, new_freq=sample_rate
            )

            # (1,L) -> (L)
            waveform = waveform.squeeze(0)

            waveform = trimmer(waveform)
            waveform = padder(waveform)

            self.waveforms[i] = waveform.cpu()

        print("Dataset loaded")
        del dataset_df

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        return self.waveforms[idx]


def compute_dataset_stats(dataset_path):
    # from http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
    stats_loader = create_dataloader(dataset_path, 100, 0)
    mean = []
    std = []
    seen = 0.0
    for batch, _ in stats_loader:
        N = batch.shape[0]
        batch = batch.view(N, 3, -1)
        for i in range(3):
            channel = batch[:, i, :]
            if seen == 0:
                mean.append(channel.mean())
                std.append(channel.std())
            else:
                new_mean = channel.mean()
                new_std = channel.std()

                tmp_mean = mean[i]

                mean[i] = seen / (seen + N) * tmp_mean + N / (seen + N) * new_mean
                std[i] = (
                    seen / (seen + N) * std[i] ** 2
                    + N / (seen + N) * new_std**2
                    + seen * N / (seen + N) ** 2 * (tmp_mean - new_std) ** 2
                )
                std[i] = np.sqrt(std[i])
        seen += N

    return mean, std


def create_distil_dataloader(
    dataset_csv: str,
    teacher_logits_path: str,
    names,
    sample_rate,
    duration,
    batch_size: int,
    num_workers: int,
    waveform_transforms=None,
    shuffle=True,
) -> tdata.DataLoader:
    dataset = DistilAudioDataset(
        dataset_csv,
        teacher_logits_path,
        names,
        sample_rate,
        duration,
        waveform_transforms,
    )
    dataloader = tdata.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=3,
    )
    return dataloader


def create_teacher_compute_dataloader(
    dataset_csv: str,
    names: list[str],
    sample_rate: int,
    duration: int,
    batch_size: int,
    num_workers: int,
) -> tdata.DataLoader:
    dataset = TeacherAudioDataset(dataset_csv, names, sample_rate, duration)
    dataloader = tdata.DataLoader(
        dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers
    )
    return dataloader


def create_dataloader(
    pkl_dir: Union[str, list[str]],
    batch_size: int,
    num_workers: int,
    waveform_transforms=None,
    spectrogram_transforms=None,
    filter_datasets=[],
    device="cpu",
) -> tdata.DataLoader:
    dataset = AudioDataset(
        pkl_dir,
        waveform_transforms=waveform_transforms,
        spectrogram_transforms=spectrogram_transforms,
        filter_datasets=filter_datasets,
    )
    dataloader = tdata.DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    return dataloader

def create_raw_dataloader(
    dataset_df: pd.DataFrame,
    names: list[str],
    sample_rate: int,
    duration: int,
    batch_size: int,
    num_workers: int,
    waveform_transforms=None,
    spectrogram_transforms=None,
    filter_datasets=[],
) -> tdata.DataLoader:
    extractor = LogMelSpectrogramExtractorModel(
        sample_rate=sample_rate, n_mels=128, length=250, duration=10
    )
    dataset = RawAudioDataset(
        dataset_df,
        names,
        sample_rate,
        duration,
        extractor,
        waveform_transforms,
        spectrogram_transforms,
        filter_datasets,
    )
    dataloader = tdata.DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    return dataloader