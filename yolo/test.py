import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
import pandas as pd

from utils.datasets import create_dataloader
from utils.general import (
    coco80_to_coco91_class,
    check_dataset,
    check_file,
    check_img_size,
    box_iou,
    non_max_suppression,
    scale_coords,
    xyxy2xywh,
    xywh2xyxy,
    clip_coords,
    set_logging,
)
from utils.loss import compute_loss
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized

from models.models import *  # noqa: F403


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, "r") as f:
        names = f.read().split("\n")
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def load_datasets(path, datasets_id):
    # Loads *.names file at 'path'
    with open(path, "r") as f:
        paths_to_datasets = f.read().split("\n")
    datasets = [list(filter(None, paths_to_datasets))[index] for index in datasets_id]
    print(f"Datasets loaded: {datasets}")
    return datasets


def test(
    data,
    weights=None,
    batch_size=16,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.6,  # for NMS
    save_json=False,
    single_cls=False,
    augment=False,
    verbose=False,
    model=None,
    dataloader=None,
    save_dir=Path(""),  # for saving images
    save_txt=False,  # for auto-labelling
    save_conf=False,
    save_incorrect=False,
    plots=True,
    log_imgs=0,
    exp_name=None,
):  # number of logged images

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        save_txt = opt.save_txt  # save *.txt labels

        # Directories
        # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        timestr = time.strftime("%Y_%m_%d-%H_%M_%S-")
        if not opt.datasets:  # just 1 dataset
            out = (
                "/media/magalhaes/output/test/"
                + opt.data[2:-5].replace("/", "_")
                + "/"
                + timestr
                + str(opt.conf_thres)
                + "/"
            )
        else:
            out = (
                "/media/magalhaes/output/test/multiple/"
                + str(opt.datasets)[1:-1].replace(", ", "_")
                + "/"
                + timestr
                + str(opt.conf_thres)
                + "/"
            )
        save_dir = Path(out)
        (save_dir / "labels" if save_txt else save_dir).mkdir(
            parents=True, exist_ok=True
        )  # make dir

        # Load model
        model = Darknet(opt.cfg).to(device)

        # load model
        try:
            ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
            ckpt["model"] = {
                k: v
                for k, v in ckpt["model"].items()
                if model.state_dict()[k].numel() == v.numel()
            }
            model.load_state_dict(ckpt["model"], strict=False)
        except:
            load_darknet_weights(model, weights[0])
        imgsz = check_img_size(imgsz, s=64)  # check img_size

    # Half
    half = device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure - basically defines the number of classes
    model.eval()
    is_coco = data.endswith("coco.yaml")  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = min(log_imgs, 50)  # ceil
    wandb = False
    try:
        import wandb  # Weights & Biases

        if not training:
            wandb.init(
                project="Vision",
                entity="magalhaes",
                group="yolov4",
                job_type="study" if opt.task == "study" else "test",
                name=exp_name,
            )
            wandb.config = {
                "batch_size": batch_size,
                "imgsz": imgsz,
                "conf_thres": conf_thres,
                "iou_thres": iou_thres,
                "single_cls": single_cls,
                "augment": augment,
            }
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = (
            model(img.half() if half else img) if device.type != "cpu" else None
        )  # run once
        if not opt.datasets:  # just 1 dataset
            path = (
                data["test"] if opt.task == "test" else data["val"]
            )  # path to val/test images
        else:  # multiple datasets
            path = load_datasets("./data/all.dataset", opt.datasets)
        dataloader = create_dataloader(
            path, imgsz, batch_size, 64, opt, pad=0.5, rect=True
        )[0]
    seen = 0
    try:
        names = model.names if hasattr(model, "names") else model.module.names
    except:
        names = load_classes(opt.names)
    coco91class = coco80_to_coco91_class()
    s = ("%20s" + "%12s" * 7) % (
        "Class",
        "Images",
        "Targets",
        "Predicted",
        "P",
        "R",
        "mAP@.5",
        "mAP@.5:.95",
    )
    p, r, f1, mp, mr, map50, map, t0, t1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, train_out = model(
                img, augment=augment
            )  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][
                    :3
                ]  # box, obj, cls

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(
                inf_out, conf_thres=conf_thres, iou_thres=iou_thres
            )
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append(
                        (
                            torch.zeros(0, niou, dtype=torch.bool),
                            torch.Tensor(),
                            torch.Tensor(),
                            tcls,
                        )
                    )
                continue

            # Append to text file
            path = Path(paths[si])
            if save_txt:
                gn = torch.tensor(shapes[si][0])[
                    [1, 0, 1, 0]
                ]  # normalization gain whwh
                x = pred.clone()
                x[:, :4] = scale_coords(
                    img[si].shape[1:], x[:, :4], shapes[si][0], shapes[si][1]
                )  # to original
                for *xyxy, conf, cls in x:
                    xywh = (
                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                        .view(-1)
                        .tolist()
                    )  # normalized xywh
                    line = (
                        (cls, *xywh, conf) if save_conf else (cls, *xywh)
                    )  # label format
                    with open(save_dir / "labels" / (path.stem + ".txt"), "a") as f:
                        f.write(("%g " * len(line)).rstrip() % line + "\n")

            # W&B logging
            if plots and len(wandb_images) < log_imgs and "schreder" in str(path):
                class_id_to_label = {k: v for k, v in enumerate(names)}
                box_data = [
                    {
                        "position": {
                            "minX": xyxy[0],
                            "minY": xyxy[1],
                            "maxX": xyxy[2],
                            "maxY": xyxy[3],
                        },
                        "class_id": int(cls),
                        "box_caption": "%s %.3f" % (names[int(cls)], conf),
                        "scores": {"class_score": conf},
                        "domain": "pixel",
                    }
                    for *xyxy, conf, cls in pred.tolist()
                    if conf > 0.01
                ]
                boxes = {
                    "predictions": {
                        "box_data": box_data,
                        "class_labels": class_id_to_label,
                    }
                }
                wandb_images.append(
                    wandb.Image(img[si], boxes=boxes, caption=path.name)
                )

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = pred[:, :4].clone()  # xyxy
                scale_coords(
                    img[si].shape[1:], box, shapes[si][0], shapes[si][1]
                )  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append(
                        {
                            "image_id": image_id,
                            "category_id": coco91class[int(p[5])]
                            if is_coco
                            else int(p[5]),
                            "bbox": [round(x, 3) for x in b],
                            "score": round(p[4], 5),
                        }
                    )

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (
                        (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    )  # prediction indices
                    pi = (
                        (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                    )  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(
                            1
                        )  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if (
                                    len(detected) == nl
                                ):  # all targets already located in image
                                    break

                # Add to incorrect file with: path, num_correct, fraction_correct, num_pred, num_targets
                if save_incorrect:
                    with open(save_dir / "incorrect.csv", "a+") as fp:
                        num_correct = correct.sum().item()
                        fraction_correct = (
                            float(num_correct) / len(correct) if len(correct) else 0
                        )
                        num_pred = len(detected)
                        num_targets = nl
                        fp.write(
                            f"{path},{num_correct},{fraction_correct},{num_pred},{num_targets}\n"
                        )

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f"test_batch{batch_i}_labels.jpg"  # filename
            plot_images(img, targets, paths, f, names)  # labels
            f = save_dir / f"test_batch{batch_i}_pred.jpg"
            plot_images(
                img, output_to_target(output, width, height), paths, f, names
            )  # predictions

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    # Ignore "other" class for overall metrics
    other_cls_id = 6
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(
            *stats, plot=plots, fname=save_dir / "precision-recall_curve.png"
        )
        # The "other" class is not considered for the "all" metrics
        p_all, r_all, ap50_all, ap_all = (
            np.concatenate((p[:other_cls_id, 0], p[other_cls_id + 1 :, 0])),
            np.concatenate((r[:other_cls_id, 0], r[other_cls_id + 1 :, 0])),
            np.concatenate((ap[:other_cls_id, 0], ap[other_cls_id + 1 :, 0])),
            np.concatenate((ap[:other_cls_id], ap[other_cls_id + 1 :])).mean(1),
        )
        mp, mr, map50, map = p_all.mean(), r_all.mean(), ap50_all.mean(), ap_all.mean()
        p, r, ap50, ap = (
            p[:, 0],
            r[:, 0],
            ap[:, 0],
            ap.mean(1),
        )  # [P, R, AP@0.5, AP@0.5:0.95]
        nt = np.bincount(
            stats[3].astype(np.int64), minlength=nc
        )  # number of targets per class
        npred = np.bincount(
            stats[2].astype(np.int64), minlength=nc
        )  # number of predictions per class
        nt_all = np.concatenate((nt[:other_cls_id], nt[other_cls_id + 1 :]))
        npred_all = np.concatenate((npred[:other_cls_id], npred[other_cls_id + 1 :]))
    else:
        nt_all = torch.zeros(1)
        npred_all = torch.zeros(1)
        nt = torch.zeros(1)
        npred = torch.zeros(1)

    # W&B logging
    if plots and wandb:
        wandb.log({"Images": wandb_images})
        wandb.log(
            {
                "Validation": [
                    wandb.Image(str(x), caption=x.name)
                    for x in sorted(save_dir.glob("test*.jpg"))
                ]
            }
        )

    # Print results
    pf = "%20s" + "%12.3g" * 7  # print format
    print(pf % ("all", seen, nt_all.sum(), npred_all.sum(), mp, mr, map50, map))
    if wandb:
        wandb.log(
            {
                "Mean Precision": mp,
                "Mean Recall": mr,
                "mAP@50": map50,
                "mAP": map,
            }
        )

    # Print results per class
    if nc > 1 and len(stats):
        data = {
            "Class": ["all"],
            "# images": [seen],
            "# targets": [nt_all.sum()],
            "# predictions": [npred_all.sum()],
            "Precision": [mp],
            "Recall": [mr],
            "mAP@.5": [map50],
            "mAP@.5:.95": [map],
        }
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], npred[c], p[i], r[i], ap50[i], ap[i]))
            data["Class"].append(names[c])
            data["# images"].append(seen)
            data["# targets"].append(nt[c])
            data["# predictions"].append(npred[c])
            data["Precision"].append(p[i])
            data["Recall"].append(r[i])
            data["mAP@.5"].append(ap50[i])
            data["mAP@.5:.95"].append(ap[i])
        if wandb and not training:
            wandb.log({"Results Table": pd.DataFrame(data)})

    # Print speeds
    t = tuple(x / seen * 1e3 for x in (t0, t1, t0 + t1)) + (
        imgsz,
        imgsz,
        batch_size,
    )  # tuple
    if not training:
        print(
            "Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g"
            % t
        )

    # Save JSON
    if save_json and len(jdict):
        w = (
            Path(weights[0] if isinstance(weights, list) else weights).stem
            if weights is not None
            else ""
        )  # weights
        anno_json = glob.glob("/media/magalhaes/coco/annotations/instances_val*.json")[
            0
        ]  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print("\nEvaluating pycocotools mAP... saving %s..." % pred_json)
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, "bbox")
            if is_coco:
                eval.params.imgIds = [
                    int(Path(x).stem) for x in dataloader.dataset.img_files
                ]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print("ERROR: pycocotools unable to run: %s" % e)

    # Return results
    if not training:
        print("Results saved to %s" % save_dir)
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test.py")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="yolov4-csp.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--data", type=str, default="data/coco.yaml", help="*.data path"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="size of each image batch"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.05, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.55, help="IOU threshold for NMS"
    )
    parser.add_argument("--task", default="val", help="'val', 'test', 'study'")
    parser.add_argument(
        "--save-incorrect", action="store_true", help="save incorrect to *.csv"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="treat as single-class dataset"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="save a cocoapi-compatible JSON results file",
    )
    parser.add_argument("--project", default="runs/test", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--cfg", type=str, default="models/yolov4-csp.cfg", help="*.cfg path"
    )
    parser.add_argument(
        "--names", type=str, default="data/coco.names", help="*.cfg path"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=int,
        help="datasets to run according to datasets.names datasets dictionary",
    )
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ["val", "test"]:  # run normally
        if ".pt" in opt.weights[0]:
            exp_name = (
                opt.weights[0].split("/")[-3]
                + "_"
                + opt.weights[0].split(".pt")[-2].split("/")[-1]
                + "_"
                + opt.data.split(".")[-2].split("/")[-1]
            )
        else:
            # .weights
            exp_name = (
                opt.weights[0].split(".")[-2].split("/")[-1]
                + "_"
                + opt.data.split(".")[-2].split("/")[-1]
            )
        test(
            opt.data,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.conf_thres,
            opt.iou_thres,
            opt.save_json,
            opt.single_cls,
            opt.augment,
            opt.verbose,
            save_txt=opt.save_txt,
            save_conf=opt.save_conf,
            save_incorrect=opt.save_incorrect,
            exp_name=exp_name,
        )

    elif opt.task == "study":  # run over a range of settings and save/plot
        for weights in opt.weights:
            f = "study_%s_%s.txt" % (
                Path(opt.data).stem,
                Path(weights).stem,
            )  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print("\nRunning %s point %s..." % (f, i))
                r, _, t = test(
                    opt.data,
                    [weights],
                    opt.batch_size,
                    i,
                    opt.conf_thres,
                    opt.iou_thres,
                    opt.save_json,
                )
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt="%10.4g")  # save
        os.system("zip -r study.zip study_*.txt")
        # utils.general.plot_study_txt(f, x)  # plot
