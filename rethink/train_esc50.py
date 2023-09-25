import os
import argparse
from typing import Optional

import torch
import torchvision.transforms as T
import torchaudio.transforms as TA_T
import torch.nn as nn

import pandas as pd
from tqdm import tqdm
import wandb

import rethink.utils as utils
import rethink.evaluate as evaluate
from rethink.preprocess import LogMelSpectrogramExtractorModel
from rethink.models import create_model, ModelEma
from rethink.dataset import create_dataloader, Roll


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

    with tqdm(total=len(data_loader)) as t:
        for batch_idx, data in enumerate(data_loader):
            inputs = data[0].to(device)
            # teacher_inputs = data[1].to(device)
            target = data[1].squeeze(1).to(device)

            # inputs = extractor(inputs)
            # teacher_inputs = extractor(teacher_inputs)

            # inputs = spec_aug(inputs)
            # teacher_inputs = spec_aug(teacher_inputs)
            N = inputs.shape[0]
            if mixup > 0.0:
                rn_indices, lam = utils.mixup(N, mixup)
                lam = lam.to(device)
                inputs = inputs * lam.reshape(N, 1, 1, 1) + inputs[rn_indices] * (
                    1.0 - lam.reshape(N, 1, 1, 1)
                )
                target = target * lam.reshape(N, 1) + target[rn_indices] * (
                    1.0 - lam.reshape(N, 1)
                )
            outputs = model(inputs)
            loss = loss_fn(outputs, target)

            logs_dict = {"Loss": loss.item()}

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            model, ema, device, train_loader, optimizer, loss_fn, mixup=mixup
        )
        logs_dict, _ = evaluate.evaluate(
            model,
            device,
            val_loader,
            id_to_class_name,
            loss_fn=loss_fn,
            epoch=epoch,
            multi_class=True,
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


def load_best_model(params, split, model):
    best_model_path = os.path.join(
        params.checkpoint_dir, wandb.run.name, f"model_best_{split}.pth"
    )
    if os.path.exists(best_model_path):
        model.load_state_dict(
            torch.load(best_model_path, map_location=torch.device("cpu"))["model"]
        )


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

    train_spec_transforms = T.Compose(
        [
            Roll((0, 0, 1), (0, 1, 2), 0, 250),
            TA_T.FrequencyMasking(freq_mask_param=10, iid_masks=True),
            TA_T.TimeMasking(time_mask_param=10, iid_masks=True),
        ]
    )
    train_waveform_transforms = None  # WaveformAugmentations()
    test_spec_transforms = T.Compose([])

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
    for split in range(params.num_folds):
        train_split = [
            f"{params.split_base}_split{i+1}.pkl"
            for i in range(params.num_folds)
            if i != split
        ]
        test_split = f"{params.split_base}_split{split+1}.pkl"
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

        model = create_model(
            params.model,
            params.num_classes,
            params.pretrained_weights,
            device,
        )

        loss_fn = nn.CrossEntropyLoss()
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

        # Load best model
        load_best_model(params, split + 1, model)

        final_logs, (logits, targets) = evaluate.evaluate(
            model, device, val_loader, id_to_class_name, eval_run=True, multi_class=True
        )
        splits_final_evals.append(final_logs)
        splits_logits.append(logits)
        splits_targets.append(targets)

    # Final log dict
    logs_dict = {}

    # Compute Cross Validation Results
    runs_tables = [log["Results Table"] for log in splits_final_evals]
    mean_res = pd.concat(runs_tables)
    mean_res = mean_res.groupby(mean_res.index).mean()
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

    wandb.run.finish()
