import os
import yaml
import argparse
from pathlib import Path

# import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
import wandb

from yolov7.models.yolo import Model, DistillationType
from yolov7.utils.general import (
    increment_path,
    init_seeds,
    colorstr,
    check_img_size,
    labels_to_class_weights,
    one_cycle,
    fitness,
)
from yolov7.utils.loss import (
    ComputeLossOTA,
    ComputeLoss,
    ComputeKDLoss,
    ComputeFineGrainedFeatureImitationLoss,
)
from yolov7.utils.torch_utils import intersect_dicts, select_device, ModelEMA
from yolov7.utils.datasets import create_dataloader
import yolov7.test as test


class EnsembleKD(nn.ModuleList):
    # Ensemble of models
    def __init__(self, modules):
        super(EnsembleKD, self).__init__(modules)

    def forward(self, x, augment=False, distillation=False):
        y = []
        features = []
        for module in self:
            y_ = module(x, augment=augment, distillation=distillation)
            if distillation:
                (y_, features_) = y_
                features.append(features_)
            y.append(y_)
        mean_y = []
        for i in range(len(y[0])):
            mean_y.append(torch.stack([x[i] for x in y]).mean(0))  # mean ensemble
        if distillation:
            mean_features = []
            for i in range(len(features[0])):
                mean_features.append(
                    torch.stack(
                        [x[i].abs().pow(2).sum(dim=1) for x in features], 0
                    ).mean(0)
                )  # mean ensemble
            return mean_y, mean_features  # inference, hidden features
        else:
            return mean_y


def create_optimizer(args: object, model: Model, hyp: dict):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, "im"):
            if hasattr(v.im, "implicit"):
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, "imc"):
            if hasattr(v.imc, "implicit"):
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, "imb"):
            if hasattr(v.imb, "implicit"):
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, "imo"):
            if hasattr(v.imo, "implicit"):
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, "ia"):
            if hasattr(v.ia, "implicit"):
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, "attn"):
            if hasattr(v.attn, "logit_scale"):
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, "q_bias"):
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, "v_bias"):
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, "relative_position_bias_table"):
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, "rbr_dense"):
            if hasattr(v.rbr_dense, "weight_rbr_origin"):
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, "weight_rbr_avg_conv"):
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, "weight_rbr_pfir_conv"):
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, "weight_rbr_1x1_kxk_idconv1"):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, "weight_rbr_1x1_kxk_conv2"):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, "weight_rbr_gconv_dw"):
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, "weight_rbr_gconv_pw"):
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, "vector"):
                pg0.append(v.rbr_dense.vector)

    if args.adam:
        optimizer = optim.Adam(
            pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999)
        )  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(
            pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True
        )

    optimizer.add_param_group(
        {"params": pg1, "weight_decay": hyp["weight_decay"]}
    )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    print(
        "Optimizer groups: %g .bias, %g conv.weight, %g other"
        % (len(pg2), len(pg1), len(pg0))
    )
    del pg0, pg1, pg2
    return optimizer


def create_dataset_dataloader(
    args, data_path, grid_size, img_size, train, hyp, num_classes
):
    # Trainloader
    dataloader, dataset = create_dataloader(
        data_path,
        img_size,
        args.batch_size,
        grid_size,
        args,
        hyp=hyp,
        augment=train,
        cache=False,
        rect=not train,
        rank=-1,
        world_size=1,
        workers=args.workers,
        image_weights=False,
        quad=False,
        pad=0.0 if train else 0.5,
        prefix=colorstr("distil: " if train else "val: "),
    )
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    num_batches = len(dataloader)  # number of batches
    assert (
        mlc < num_classes
    ), "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g" % (
        mlc,
        num_classes,
        args.data,
        num_classes - 1,
    )
    return dataloader, dataset, num_batches


def load_teacher(
    teacher_weights: str,
    teacher_cfg: str,
    num_classes: int,
    device: torch.device,
    distillation_type: DistillationType,
) -> Model:
    ckpt = torch.load(teacher_weights, map_location=device)  # load checkpoint
    model = Model(
        teacher_cfg,
        ch=3,
        nc=num_classes,
        anchors=None,
        distillation_type=distillation_type,
    ).to(device)
    exclude = ["anchor"]  # exclude keys
    state_dict = ckpt["model"].float().state_dict()  # to FP32
    state_dict = intersect_dicts(
        state_dict, model.state_dict(), exclude=exclude
    )  # intersect
    model.load_state_dict(state_dict, strict=False)  # load

    # Freeze teacher model
    for param in model.parameters():
        param.requires_grad = False

    print(
        "Transferred %g/%g items from %s"
        % (len(state_dict), len(model.state_dict()), teacher_weights)
    )  # report

    model.train()
    return model


def train_and_test(
    args: object,
    model: Model,
    teacher: Model,
    dataloader: DataLoader,
    testloader: DataLoader,
    optimizer: optim.Optimizer,
    hyp: dict,
    num_batches: int,
    accumulate: int,
    num_classes: int,
    device: torch.device,
):
    # t0 = time.time()
    best_fitness = 0.0
    epochs = args.epochs
    nw = max(
        round(hyp["warmup_epochs"] * num_batches), 1000
    )  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scaler = GradScaler(enabled=True)
    compute_loss = ComputeLoss(model)  # init loss class
    compute_loss_ota = ComputeLossOTA(model)  # init loss class
    distillation_pred_loss_fn = ComputeKDLoss(model)
    fgfi_loss_fn = ComputeFineGrainedFeatureImitationLoss(model, 0.5)
    transfer_loss_fn = nn.L1Loss()

    ema = ModelEMA(model)

    # Feature projectors
    projectors = ()
    if model.distillation_type == DistillationType.BACKBONE_NECK_HEAD:
        projectors = (
            nn.Conv2d(512, 1280, 1, 1).to(device),
            nn.Conv2d(256, 640, 1, 1).to(device),
            nn.Conv2d(128, 320, 1, 1).to(device),
            nn.Conv2d(64, 160, 1, 1).to(device),
            nn.Conv2d(128, 320, 1, 1).to(device),
            nn.Conv2d(256, 640, 1, 1).to(device),
        )
    elif model.distillation_type == DistillationType.NECK:
        projectors = (
            nn.Conv2d(256, 640, 1, 1).to(device),
            nn.Conv2d(128, 320, 1, 1).to(device),
            nn.Conv2d(64, 160, 1, 1).to(device),
        )
    elif model.distillation_type == DistillationType.HEAD:
        projectors = (
            nn.Conv2d(64, 160, 1, 1).to(device),
            nn.Conv2d(128, 320, 1, 1).to(device),
            nn.Conv2d(256, 640, 1, 1).to(device),
        )

    # Add projectors to optimizer
    for projector in projectors:
        optimizer.add_param_group(
            {"params": projector.parameters(), "weight_decay": hyp["weight_decay"]}
        )

    lf = one_cycle(1, hyp["lrf"], epochs)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(epochs):
        model.train()

        mloss = torch.zeros(4, device=device)  # mean losses
        kd_loss = 0.0
        print(
            ("\n" + "%10s" * 10)
            % (
                "Epoch",
                "gpu_mem",
                "kd",
                "transfer",
                "box",
                "obj",
                "cls",
                "total",
                "labels",
                "img_size",
            )
        )
        pbar = tqdm(enumerate(dataloader), total=num_batches)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, _, _) in pbar:
            ni = (
                i + num_batches * epoch
            )  # number integrated batches (since train start)
            imgs = (
                imgs.to(device, non_blocking=True).float() / 255.0
            )  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(
                    1, np.interp(ni, xi, [1, 64 / args.batch_size]).round()
                )
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(
                        ni,
                        xi,
                        [
                            hyp["warmup_bias_lr"] if j == 2 else 0.0,
                            x["initial_lr"] * lf(epoch),
                        ],
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(
                            ni, xi, [hyp["warmup_momentum"], hyp["momentum"]]
                        )

            # Forward
            with autocast(enabled=True):
                pred, distillation_out = model(imgs, distillation=True)  # forward
                soft_targets, teacher_distillation_out = teacher(
                    imgs, distillation=True
                )
                # HEAD is always the last element of distillation_out
                # we need to remove it if we don't want to use it
                # NECK and HEAD share one layer
                if model.distillation_type in [
                    DistillationType.BACKBONE_NECK,
                    DistillationType.NECK,
                ]:
                    distillation_out = distillation_out[:-2]
                    teacher_distillation_out = teacher_distillation_out[:-2]
                elif model.distillation_type in [DistillationType.BACKBONE]:
                    distillation_out = distillation_out[:-3]
                    teacher_distillation_out = teacher_distillation_out[:-3]

                for i in range(len(projectors)):
                    distillation_out[i] = projectors[i](distillation_out[i])

                transfer_loss = torch.zeros(1, device=device)
                if epoch > -1:
                    detection_loss, loss_items = compute_loss_ota(
                        pred, targets.to(device), imgs
                    )  # loss scaled by batch_size
                    # calculate kd loss for each classifier
                    if args.kd_ratio > 0.0:
                        kd_loss = 0.0
                        if model.distillation_type != DistillationType.NONE:
                            transfer_loss, fgfi_mask = fgfi_loss_fn(
                                distillation_out,
                                teacher_distillation_out,
                                soft_targets,
                                targets.to(device),
                            )
                            kd_loss = distillation_pred_loss_fn(
                                pred, soft_targets, fgfi_mask
                            )[0]
                        loss = (1 - args.kd_ratio) * detection_loss + args.kd_ratio * (
                            kd_loss + transfer_loss
                        )
                    else:
                        loss = detection_loss
                else:
                    for i in range(len(teacher_distillation_out)):
                        transfer_loss += transfer_loss_fn(
                            distillation_out[i] / F.normalize(distillation_out[i]),
                            teacher_distillation_out[i]
                            / F.normalize(distillation_out[i]),
                        )
                    transfer_loss /= len(teacher_distillation_out)
                    loss = transfer_loss
                    loss_items = torch.zeros(4, device=device)
            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = "%.3gG" % (
                torch.cuda.memory_reserved(device) / 1e9
                if torch.cuda.is_available()
                else 0
            )  # (GB)
            s = ("%10s" * 2 + "%10.4g" * 8) % (
                "%g/%g" % (epoch, epochs - 1),
                mem,
                kd_loss,
                transfer_loss,
                *mloss,
                targets.shape[0],
                imgs.shape[-1],
            )
            pbar.set_description(s)

            # end batch ------------------------------------------------------
        # end epoch ----------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # mAP
        ema.update_attr(
            model,
            include=["yaml", "nc", "hyp", "gr", "names", "stride", "class_weights"],
        )
        results, _, _ = test.test(
            data_dict,
            batch_size=args.batch_size * 2,
            imgsz=img_size_test,
            model=ema.ema,
            single_cls=False,
            dataloader=testloader,
            save_dir=args.save_dir,
            verbose=False,
            plots=False,
            wandb_logger=None,
            compute_loss=compute_loss,
            is_coco=True,
        )

        # Log
        tags = [
            "train/kd_loss",
            "train/transfer_loss",
            "train/box_loss",
            "train/obj_loss",
            "train/cls_loss",  # train loss
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP_0.5",
            "metrics/mAP_0.5:0.95",
            "val/box_loss",
            "val/obj_loss",
            "val/cls_loss",  # val loss
            "x/lr0",
            "x/lr1",
            "x/lr2",
        ]  # params
        for x, tag in zip(
            [kd_loss] + [transfer_loss] + list(mloss[:-1]) + list(results) + lr, tags
        ):
            wandb.log({tag: x})  # W&B

        # Update best mAP
        fi = fitness(
            np.array(results).reshape(1, -1)
        )  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        ckpt = {
            "epoch": epoch,
            "best_fitness": best_fitness,
            "model": deepcopy(model).half(),
            "ema": deepcopy(ema.ema).half(),
            "updates": ema.updates,
            "optimizer": optimizer.state_dict(),
        }

        # Save last, best and delete
        weights_dir = args.save_dir / "weights"
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(ckpt, weights_dir / "last.pt")
        if best_fitness == fi:
            torch.save(ckpt, weights_dir / "best.pt")
        del ckpt

    # end training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distillate a model on Audioset from Ensemble of PaSST models"
    )
    parser.add_argument("--hyp", default="data/hyp.scratch.tiny.yaml", type=str)
    parser.add_argument(
        "--teacher_weights", type=str, default="yolo7.pt", help="initial weights path"
    )
    parser.add_argument(
        "--teacher_cfg",
        type=str,
        default="cfg/training/yolov7x.yaml",
        help="model.yaml path",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="cfg/training/yolov7-tiny.yaml",
        help="model.yaml path",
    )
    parser.add_argument(
        "--data", type=str, default="data/coco.yaml", help="data.yaml path"
    )
    parser.add_argument(
        "--kd-ratio",
        type=float,
        default=0.8,
        help="ratio of distillation loss in total loss",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--batch-size", type=int, default=16, help="total batch size for all GPUs"
    )
    parser.add_argument(
        "--adam", action="store_true", help="use torch.optim.Adam() optimizer"
    )
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="[train, test] image sizes",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="maximum number of dataloader workers"
    )
    parser.add_argument(
        "--project",
        default="./runs/distillation",
        help="save to project/name",
    )
    parser.add_argument("--exp_name", default="distil_yolov7", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()

    # Add args that are not configurable in this script
    args.single_cls = False

    # Load Hyperparameters
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Load dict with data configs
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = args.data.endswith("coco.yaml")

    device = select_device(args.device)

    args.save_dir = Path(
        increment_path(Path(args.project) / args.exp_name, exist_ok=True)
    )
    os.makedirs(args.save_dir, exist_ok=True)

    # Save run settings
    with open(args.save_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(args.save_dir / "args.yaml", "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)

    num_classes = int(data_dict["nc"])
    names = data_dict["names"]  # class names
    assert len(names) == num_classes, "%g names found for nc=%g dataset in %s" % (
        len(names),
        num_classes,
        args.data,
    )  # check

    # Seed
    init_seeds()

    distillation_type = DistillationType.HEAD

    # Create student
    model = Model(
        args.cfg,
        ch=3,
        nc=num_classes,
        anchors=hyp.get("anchors"),
        distillation_type=distillation_type,
    ).to(device)
    model.half().float()
    ckpt = torch.load(
        "/media/magalhaes/vision/pretrained/yolov7-tiny.pt", map_location=device
    )
    exclude = ["anchor"]  # exclude keys
    state_dict = ckpt["model"].float().state_dict()  # to FP32
    state_dict = intersect_dicts(
        state_dict, model.state_dict(), exclude=exclude
    )  # intersect
    model.load_state_dict(state_dict, strict=False)  # load

    # Create teacher
    # teacher = EnsembleKD(
    #     [
    #         load_teacher(
    #             "/media/magalhaes/vision/pretrained/yolov7_training.pt",
    #             "./cfg/training/yolov7.yaml",
    #             num_classes,
    #             device,
    #             distillation_type,
    #         ),
    #         load_teacher(
    #             "/media/magalhaes/vision/pretrained/yolov7x_training.pt",
    #             "./cfg/training/yolov7x.yaml",
    #             num_classes,
    #             device,
    #             distillation_type,
    #         ),
    #         # load_teacher("/media/magalhaes/vision/pretrained/yolov7-w6_training.pt", "./cfg/training/yolov7-w6.yaml", num_classes, device, distillation_type),
    #         # load_teacher("/media/magalhaes/vision/pretrained/yolov7-e6e_training.pt", "./cfg/training/yolov7-e6e.yaml", num_classes, device, distillation_type),
    #     ]
    # )
    teacher = load_teacher(
        args.teacher_weights,
        args.teacher_cfg,
        num_classes,
        device,
        distillation_type,
    )

    # Create datasets and dataloaders
    train_path = data_dict["train"]
    test_path = data_dict["val"]

    grid_size = max(int(model.stride.max()), 32)
    img_size, img_size_test = [
        check_img_size(x, grid_size) for x in args.img_size
    ]  # verify imgsz are gs-multiples

    train_dataloader, train_dataset, num_batches = create_dataset_dataloader(
        args, train_path, grid_size, img_size, True, hyp, num_classes
    )
    test_dataloader, _, _ = create_dataset_dataloader(
        args, test_path, grid_size, img_size_test, False, hyp, num_classes
    )

    # Optimizer and scheduler
    hyp["lr0"] = hyp["lr0"] / (1 - args.kd_ratio)
    nominal_batch_size = 64  # nominal batch size
    accumulate = max(
        round(nominal_batch_size / args.batch_size), 1
    )  # accumulate loss before optimizing
    hyp["weight_decay"] *= (
        args.batch_size * accumulate / nominal_batch_size
    )  # scale weight_decay
    optimizer = create_optimizer(args, model, hyp)

    # Model parameters
    num_detection_layers = model.model[-1].nl
    hyp["box"] *= 3.0 / num_detection_layers  # scale to layers
    hyp["cls"] *= (
        num_classes / 80.0 * 3.0 / num_detection_layers
    )  # scale to classes and layers
    hyp["obj"] *= (
        (img_size / 640) ** 2 * 3.0 / num_detection_layers
    )  # scale to image size and layers
    model.nc = num_classes  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = (
        labels_to_class_weights(train_dataset.labels, num_classes).to(device)
        * num_classes
    )  # attach class weights
    model.names = names

    wandb.init(
        config=hyp,
        project="Vision",
        group="yolov7",
        job_type="distil",
        name=args.exp_name,
    )

    train_and_test(
        args,
        model,
        teacher,
        train_dataloader,
        test_dataloader,
        optimizer,
        hyp,
        num_batches,
        accumulate,
        num_classes,
        device,
    )

    wandb.run.finish()
