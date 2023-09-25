import os
import argparse
from itertools import permutations
from typing import Optional

import torch
import torchvision.transforms as T
import torchaudio.transforms as TA_T
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

import rethink.utils as utils
import rethink.evaluate as evaluate
from rethink.preprocess import LogMelSpectrogramExtractorModel
from rethink.models import create_model, ModelEma
from rethink.dataset import create_dataloader, Roll, create_raw_dataloader

from audiomentations import PitchShift, AirAbsorption, ClippingDistortion, Gain, AddGaussianNoise, TimeStretch, LowPassFilter, Compose


def train(
    model: nn.Module,
    ema: Optional[ModelEma],
    device: str,
    data_loader,
    optimizer,
    loss_fn,
    mixup=0.0,
    extractor=None,
):
    model.train()
    loss_avg = utils.RunningAverage()

    # TODO dont hardcode this
    spec_aug = T.Compose(
        [
            Roll((0, 0, 1), (0, 1, 2), 0, 250),
            TA_T.FrequencyMasking(freq_mask_param=20, iid_masks=True),
            TA_T.TimeMasking(time_mask_param=20, iid_masks=True),
        ]
    )

    with tqdm(total=len(data_loader)) as t:
        for batch_idx, data in enumerate(data_loader):
            teacher_inputs = None
            if ema is not None:
                inputs = data[0].to(device)
                teacher_inputs = data[1].to(device)
                target = data[2].squeeze(1).to(device)
            else:
                inputs = data[0].to(device)
                target = data[1].squeeze(1).to(device)

            if extractor is not None:
                #TODO waveform augmentations need to be applied here
                
                inputs = extractor(inputs)
                if teacher_inputs is not None:
                    teacher_inputs = extractor(teacher_inputs)
                    teacher_inputs = spec_aug(teacher_inputs)
                inputs = spec_aug(inputs)

            N = inputs.shape[0]
            rn_indices, lam = None, None
            if mixup > 0.0:
                rn_indices, lam = utils.mixup(N, mixup)
                lam = lam.to(device)
                inputs = inputs * lam.reshape(N, 1, 1, 1) + inputs[rn_indices] * (
                    1.0 - lam.reshape(N, 1, 1, 1)
                )

                outputs = model(inputs)
                target = target * lam.reshape(N, 1) + target[rn_indices] * (
                    1.0 - lam.reshape(N, 1)
                )
            else:
                outputs = model(inputs)

            loss = loss_fn(outputs, target)

            logs_dict = {"Loss": loss.item()}

            if ema is not None and teacher_inputs is not None:
                if mixup > 0.0:
                    teacher_inputs = teacher_inputs * lam.reshape(
                        N, 1, 1, 1
                    ) + teacher_inputs[rn_indices] * (1.0 - lam.reshape(N, 1, 1, 1))
                sd_loss = F.binary_cross_entropy_with_logits(
                    outputs, ema.module(teacher_inputs).sigmoid()
                )
                logs_dict["Self Distillation Loss"] = sd_loss.item()
                loss = 0.9 * loss + 0.1 * sd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ema is not None:
                ema.update(model)

            if batch_idx % 10 == 0:
                wandb.log(logs_dict)

            loss_avg.update(loss.item())

            t.set_postfix(loss="{:05.3f}".format(loss_avg()))
            t.update()
    return loss_avg()


def train_and_evaluate(
    model,
    device,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    params,
    split,
    id_to_class_name,
    mixup=0.0,
    scheduler=None,
    extractor=None,
):
    best = 0
    ema = ModelEma(model, device=device) if params.self_distillation else None

    for epoch in range(params.epochs):
        is_best = False

        print(f"Epoch {epoch}/{params.epochs}")
        avg_loss = train(
            model,
            ema,
            device,
            train_loader,
            optimizer,
            loss_fn,
            mixup=mixup,
            extractor=extractor,
        )
        logs_dict, _ = evaluate.evaluate(
            model,
            device,
            val_loader,
            id_to_class_name,
            loss_fn=loss_fn,
            epoch=epoch,
            extractor=extractor,
        )

        print(f"Loss:{avg_loss}")

        if scheduler:
            scheduler.step()
            logs_dict["Learning Rate"] = scheduler.get_last_lr()

        if logs_dict["F1 score"] > best:
            best = logs_dict["F1 score"]
            is_best = True

        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            split,
            os.path.join(params.checkpoint_dir, wandb.run.name),
        )

        wandb.log(logs_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using Rethinking CNN Models for Audio Classification approach"
    )
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()
    params = utils.Params(args.config_path)
    device = torch.device(params.device if torch.cuda.is_available() else "cpu")

    exp_name = args.exp_name if args.exp_name is not None else params.exp_name

    wandb.init(
        config=params,
        resume="allow",
        project="Audio",
        group="rethink",
        job_type="train",
        name=exp_name,
    )

    spec_transforms_list = []

    # Read the parameters for spec_transforms
    if params.spec_transforms:
        for transform in params.spec_transforms:
            if transform == "Roll":
                spec_transforms_list.append(Roll((0, 0, 1), (0, 1, 2), 0, 250))
            elif transform == "FreqMasking":
                spec_transforms_list.append(
                    TA_T.FrequencyMasking(freq_mask_param=20, iid_masks=True)
                )
            elif transform == "TimeMasking":
                spec_transforms_list.append(
                    TA_T.TimeMasking(time_mask_param=20, iid_masks=True)
                )                

    train_spec_transforms = T.Compose(spec_transforms_list)

    train_waveform_transforms = []
    
    if params.waveform_transforms:
        for transform in params.waveform_transforms:
            if transform == "PitchShift":
                train_waveform_transforms.append(PitchShift(min_semitones=-4, max_semitones=4, p=0.5))
            elif transform == "AirAbsorption":
                train_waveform_transforms.append(AirAbsorption(min_distance=10.0, max_distance=50.0, p=0.5))
            elif transform == "ClippingDistortion":
                train_waveform_transforms.append(ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=40, p=0.5))
            elif transform == "Gain":
                train_waveform_transforms.append(Gain(min_gain_db=-15, max_gain_db=15, p=0.5))
            elif transform == "AddGaussianNoise":
                train_waveform_transforms.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5))
            elif transform == "TimeStretch":
                train_waveform_transforms.append(TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5))
            elif transform == "LowPassFilter":
                train_waveform_transforms.append(LowPassFilter(min_cutoff_freq=150.0, max_cutoff_freq=7500.0, p=0.5))

    train_waveform_transforms = Compose(train_waveform_transforms)


    test_spec_transforms = T.Compose([])

    if params.precomputed_spec or params.from_waveform:
        extractor = None
    else:
        extractor = LogMelSpectrogramExtractorModel(
            sample_rate=22050, n_mels=128, length=250, duration=10
        ).to(device)

    id_to_class_name = utils.load_id_to_class_name(params.names)

    splits_final_evals = []
    splits_final_schreder_evals = []
    splits_logits = []
    splits_targets = []
    splits_schreder_logits = []
    splits_schreder_targets = []
    folds_indexes = list(range(params.num_folds))
    for split, train_indexes in enumerate(
        permutations(folds_indexes, params.num_folds - 1)
    ):
        test_indexes = [i for i in folds_indexes if i not in train_indexes]

        if params.full_train:
            train_split = [f"{params.split_base}_split{i+1}.pkl" for i in train_indexes]
            test_split = [f"{params.split_base}_split{train_indexes[-1]+1}.pkl"]
        else:
            train_split = [f"{params.split_base}_split{i+1}.pkl" for i in train_indexes]
            test_split = [f"{params.split_base}_split{i+1}.pkl" for i in test_indexes]
        
        if hasattr(params, 'from_waveform') and params.from_waveform:
            print("Loading from waveform")

            # Load the dataset
            dataset_df = pd.read_csv(params.dataset_csv)

            # Filter the dataset by the splits, using the split column which is "splitX"
            train_split_names = [f"split{i+1}" for i in train_indexes]
            train_df = dataset_df[dataset_df["split"].isin(train_split_names)]
            test_split_names = [f"split{i+1}" for i in test_indexes]
            test_df = dataset_df[dataset_df["split"].isin(test_split_names)]

            names = ["person","bicycle","car","motorcycle","siren","bus","truck"]

            train_loader = create_raw_dataloader(
                train_df,
                names,
                params.sample_rate,
                10,
                params.batch_size,
                params.num_workers,
                waveform_transforms=train_waveform_transforms,
                spectrogram_transforms=train_spec_transforms,
            )
            val_loader = create_raw_dataloader(
                test_df,
                names,
                params.sample_rate,
                10,
                params.batch_size,
                params.num_workers,
                spectrogram_transforms=test_spec_transforms,
            )
            schreder_val_loader = create_raw_dataloader(
                test_df,
                names,
                params.sample_rate,
                10,
                params.batch_size,
                params.num_workers,
                spectrogram_transforms=test_spec_transforms,
                filter_datasets=[
                    "22_9_2022",
                    "16_12_2022",
                    "18_12_2022",
                    "21_12_2022",
                    "15_12_2022",
                    "27_01_2023",
                    "20_03_2023",
                    "11_04_2023",
                    "13_04_2023",
                ],
            )
        else:
            train_loader = create_dataloader(
                train_split,
                params.batch_size,
                params.num_workers,
                spectrogram_transforms=train_spec_transforms,
                waveform_transforms=train_waveform_transforms,
            )
            val_loader = create_dataloader(
                test_split,
                params.batch_size,
                params.num_workers,
                spectrogram_transforms=test_spec_transforms,
            )
            schreder_val_loader = create_dataloader(
                test_split,
                params.batch_size,
                params.num_workers,
                spectrogram_transforms=test_spec_transforms,
                filter_datasets=[
                    "22_9_2022",
                    "16_12_2022",
                    "18_12_2022",
                    "21_12_2022",
                    "15_12_2022",
                    "27_01_2023",
                    "20_03_2023",
                    "11_04_2023",
                    "13_04_2023",
                ],
            )


        model = create_model(
            params.model,
            params.num_classes,
            params.pretrained_weights,
            device,
        )

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params.lr, weight_decay=params.weight_decay
        )

        scheduler = None
        if params.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, int(params.epochs * 0.3), gamma=0.1
            )

        train_and_evaluate(
            model,
            device,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            params,
            split + 1,
            id_to_class_name,
            params.mixup,
            scheduler,
            extractor=extractor,
        )
        utils.load_best_model(params, split + 1, wandb.run.name, model)
        final_logs, (logits, targets) = evaluate.evaluate(
            model,
            device,
            val_loader,
            id_to_class_name,
            eval_run=True,
            extractor=extractor,
        )
        splits_final_evals.append(final_logs)
        splits_logits.append(logits)
        splits_targets.append(targets)
        schreder_final_logs, _ = evaluate.evaluate(
            model,
            device,
            schreder_val_loader,
            id_to_class_name,
            eval_run=True,
            schreder_data=True,
            extractor=extractor,
        )
        splits_final_schreder_evals.append(schreder_final_logs)
        splits_schreder_logits.append(logits)
        splits_schreder_targets.append(targets)

        # Full train uses all splits and stops after the first iteration
        if params.full_train:
            break
    # Final log dict
    logs_dict = {}

    # Create the Calibration Plots
    splits_schreder_logits = np.concatenate(splits_schreder_logits)
    splits_schreder_targets = np.concatenate(splits_schreder_targets)
    try:
        reliability_diagram_path, ece_path = evaluate.create_calibration_plots(
            splits_schreder_logits,
            splits_schreder_targets,
            list(id_to_class_name.values()),
            os.path.join(params.checkpoint_dir, wandb.run.name),
        )
        logs_dict["Reliability Diagram (Schreder data)"] = wandb.Image(
            reliability_diagram_path
        )
        logs_dict["Expected Calibration Error Plot (Schreder data)"] = wandb.Image(
            ece_path
        )
    except Exception as e:
        print(e)

    splits_logits = np.concatenate(splits_logits)
    splits_targets = np.concatenate(splits_targets)
    reliability_diagram_path, ece_path = evaluate.create_calibration_plots(
        splits_logits,
        splits_targets,
        list(id_to_class_name.values()),
        os.path.join(params.checkpoint_dir, wandb.run.name),
    )
    logs_dict["Reliability Diagram"] = wandb.Image(reliability_diagram_path)
    logs_dict["Expected Calibration Error Plot"] = wandb.Image(ece_path)

    # Compute Cross Validation Results
    if params.full_train:
        logs_dict["Results Table"] = splits_final_evals[0]["Results Table"]
    else:
        runs_tables = [log["Results Table"] for log in splits_final_evals]
        mean_res = pd.concat(runs_tables)
        mean_res = mean_res.groupby(mean_res.index).mean(numeric_only=True)
        mean_res.iloc[:, 0:3] = mean_res.iloc[:, 0:3].mul(params.num_folds)
        # sometimes the model doesn't find some classes
        # So, construct table from the most complete list of classes
        class_list = (
            list(runs_tables[1]["Class"])
            if len(runs_tables[1].index) > len(runs_tables[0].index)
            else list(runs_tables[0]["Class"])
        )
        mean_res.insert(0, "Class", class_list)
        wandb.log({"Cross Val Results Table": mean_res})

        runs_tables = [
            log["Schreder Results Table"] for log in splits_final_schreder_evals
        ]
        mean_res = pd.concat(runs_tables)
        mean_res = mean_res.groupby(mean_res.index).mean(numeric_only=True)
        mean_res.iloc[:, 0:3] = mean_res.iloc[:, 0:3].mul(params.num_folds)
        # sometimes the model doesn't find some classes
        # So, construct table from the most complete list of classes
        class_list = (
            list(runs_tables[1]["Class"])
            if len(runs_tables[1].index) > len(runs_tables[0].index)
            else list(runs_tables[0]["Class"])
        )
        mean_res.insert(0, "Class", class_list)
        logs_dict["Schreder Cross Val Results Table"] = mean_res

    wandb.log(logs_dict)
    wandb.run.finish()
