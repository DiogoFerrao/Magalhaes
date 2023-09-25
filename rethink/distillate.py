import os
import argparse

import torch
import torchvision.transforms as T
import torchaudio.transforms as TA_T
import torch.nn as nn

from tqdm import tqdm
import wandb

import rethink.utils as utils
from rethink.models import create_model
from rethink.dataset import create_distil_dataloader, Roll
from rethink.preprocess import LogMelSpectrogramExtractorModel, MelSTFT
from rethink.utils import exp_warmup_linear_down, mixup
import rethink.passt.model as passt


def train(
    model: nn.Module,
    teacher_model: nn.Module,
    device: str,
    data_loader,
    optimizer,
    loss_fn,
    distil_loss_fn,
    mel_extractor,
    teacher_mel_extractor,
    kd_lambda,
    mixup_alpha,
    train_spec_augmentations,
):
    model.train()
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(data_loader)) as t:
        for batch_idx, data in enumerate(data_loader):
            x, y = data
            x = x.to(device)
            y = y.squeeze(1).to(device)
            N = x.shape[0]

            teacher_x = teacher_mel_extractor(x).unsqueeze(1)
            x = mel_extractor(x)

            if train_spec_augmentations is not None:
                x = train_spec_augmentations(x)
                teacher_x = train_spec_augmentations(teacher_x)

            rn_indices, lam = mixup(N, mixup_alpha)
            if mixup_alpha:
                lam = lam.to(device)
                x = x * lam.reshape(N, 1, 1, 1) + x[rn_indices] * (
                    1.0 - lam.reshape(N, 1, 1, 1)
                )
                teacher_x = teacher_x * lam.reshape(N, 1, 1, 1) + teacher_x[
                    rn_indices
                ] * (1.0 - lam.reshape(N, 1, 1, 1))
                y_hat = model(x.to(device))
                y_mix = y * lam.reshape(N, 1) + y[rn_indices] * (
                    1.0 - lam.reshape(N, 1)
                )

                samples_loss = loss_fn(y_hat, y_mix)
            else:
                y_hat = model(x)
                samples_loss = loss_fn(y_hat, y)

            # hard label loss
            label_loss = samples_loss.mean()

            # distillation loss
            if kd_lambda > 0:
                with torch.no_grad():
                    teacher_logits_target = teacher_model(teacher_x)[0].sigmoid()

                if mixup_alpha:
                    teacher_logits_target_mix = teacher_logits_target * lam.reshape(
                        N, 1
                    ) + teacher_logits_target[rn_indices] * (1.0 - lam.reshape(N, 1))
                    soft_targets_loss = distil_loss_fn(
                        y_hat, teacher_logits_target_mix
                    ).mean()
                else:
                    soft_targets_loss = distil_loss_fn(
                        y_hat, teacher_logits_target
                    ).mean()  # mean since reduction="none"

                # weighting losses
                label_loss = kd_lambda * label_loss
                soft_targets_loss = (1 - kd_lambda) * soft_targets_loss
            else:
                soft_targets_loss = torch.tensor(
                    0.0, device=label_loss.device, dtype=label_loss.dtype
                )

            # total loss is sum of lambda-weighted label and distillation loss
            loss = label_loss + soft_targets_loss

            loss_avg.update(loss.item())

            t.set_postfix(loss="{:05.3f}".format(loss_avg()))
            t.update()

            # append training statistics
            if batch_idx % 10 == 0:
                wandb.log(
                    {
                        "Loss": loss.detach().cpu().numpy(),
                        "Label Loss": label_loss.detach().cpu().numpy(),
                        "Distillation Loss": soft_targets_loss.detach().cpu().numpy(),
                    }
                )

            # Update Model
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return loss_avg()


def train_and_evaluate(
    params,
    model,
    teacher_model,
    device,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    distil_loss_fn,
    mel_extractor,
    teacher_mel_extractor,
    kd_lambda,
    mixup_alpha,
    id_to_class_name,
    scheduler=None,
    train_spec_augmentations=None,
    test_spec_augmentations=None,
):

    for epoch in range(params.epochs):
        print(f"Epoch {epoch}/{params.epochs}")
        avg_loss = train(
            model,
            teacher_model,
            device,
            train_loader,
            optimizer,
            loss_fn,
            distil_loss_fn,
            mel_extractor,
            teacher_mel_extractor,
            kd_lambda,
            mixup_alpha,
            train_spec_augmentations,
        )
        # logs_dict = evaluate.evaluate(
        #     model,
        #     device,
        #     val_loader,
        #     id_to_class_name,
        #     mel_extractor=mel_extractor,
        #     loss_fn=F.binary_cross_entropy_with_logits,
        #     epoch=epoch,
        #     spec_transforms=test_spec_augmentations,
        # )
        logs_dict = {}

        print(f"Loss:{avg_loss}")

        if scheduler:
            scheduler.step()
            logs_dict["Learning Rate"] = scheduler.get_last_lr()

        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            False,
            0,
            os.path.join(params.checkpoint_dir, wandb.run.name),
        )

        wandb.log(logs_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distillate a model on Audioset from Ensemble of PaSST models"
    )
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()
    params = utils.Params(args.config_path)
    device = torch.device(params.device if torch.cuda.is_available() else "cpu")

    os.makedirs(params.checkpoint_dir, exist_ok=True)

    names = []
    for line in open(params.names).readlines():
        names.append(line.strip())

    train_waveform_transforms = T.Compose(
        [
            Roll((1,), (0,), 0, 220500),
        ]
    )
    train_spec_transforms = T.Compose(
        [
            TA_T.FrequencyMasking(freq_mask_param=40, iid_masks=True),
            TA_T.TimeMasking(time_mask_param=40, iid_masks=True),
        ]
    )
    test_spec_transforms = T.Compose([])

    train_loader = create_distil_dataloader(
        params.train_dataset_csv,
        params.teacher_logits_path,
        names,
        params.sample_rate,
        params.duration,
        params.batch_size,
        params.num_workers,
        train_waveform_transforms,
    )
    test_loader = None
    # We are currently using all data for KD
    # test_loader = create_distil_dataloader(
    #     params.test_dataset_csv,
    #     None,
    #     names,
    #     params.sample_rate,
    #     params.duration,
    #     params.batch_size,
    #     params.num_workers,
    # )

    exp_name = args.exp_name if args.exp_name is not None else params.exp_name

    wandb.init(
        config=params,
        resume="allow",
        project="Audio",
        group="rethink",
        job_type="distil",
        name=exp_name,
    )

    model = create_model(params.model, params.num_classes, None, params.device)
    model.to(device)

    teacher_ensemble = passt.get_ensemble_model(
        [
            ("passt_s_swa_p16_128_ap476", 10, 10),
            ("passt_s_swa_p16_s14_128_ap471", 14, 14),
            ("passt_s_swa_p16_s12_128_ap473", 12, 12),
        ]
    )
    teacher_ensemble.eval()
    teacher_ensemble.to(device)

    teacher_mel_extractor = MelSTFT(
        n_mels=params.n_mels, sr=params.sample_rate, device=device
    )
    teacher_mel_extractor.to(device)
    mel_extractor = LogMelSpectrogramExtractorModel(
        params.sample_rate,
        params.n_mels,
        params.image_length,
        params.duration,
        export=True,
    )
    mel_extractor.to(device)

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    distil_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=params.max_lr)

    scheduler = None
    if params.scheduler:
        schedule_lambda = exp_warmup_linear_down(
            params.warmup_len,
            params.ramp_down_len,
            params.ramp_down_start,
            params.last_lr_value,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)

    id_to_class_name = utils.load_id_to_class_name(params.names)

    train_and_evaluate(
        params,
        model,
        teacher_ensemble,
        device,
        train_loader,
        test_loader,
        optimizer,
        loss_fn,
        distil_loss_fn,
        mel_extractor,
        teacher_mel_extractor,
        params.kd_lambda,
        params.mixup_alpha,
        names,
        scheduler,
        train_spec_transforms,
        test_spec_transforms,
    )

    wandb.run.finish()
