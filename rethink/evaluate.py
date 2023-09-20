import os
import argparse
import datetime

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import wandb
import wandb.plot

from rethink.models import create_model
from rethink.dataset import create_dataloader
import rethink.utils as utils


def marginal_calibration_error(predictions, targets):
    # √(Σ(1/k E[pred - targets]**2))
    return torch.sqrt(
        torch.mean(torch.mean(torch.pow(predictions - targets, 2), dim=0), dim=0)
    )


def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    id_to_class_name: dict,
    eval_run=False,
    schreder_data=False,
    loss_fn=None,
    epoch=0,
    conf_thres=0.5,
    spec_transforms=None,
    extractor=None,
    multi_class=False,
) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    predictions = []
    logits = []
    targets = []
    losses = []
    logs_dict = {}
    total_audioclips = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            inputs = x.to(device)
            y = y.squeeze(1).cpu()

            if extractor is not None:
                inputs = extractor(inputs)

            if spec_transforms is not None:
                inputs = spec_transforms(inputs)

            outputs = model(inputs).cpu()
            if loss_fn is not None:
                losses.append(loss_fn(outputs, y).item())

            if multi_class:
                outputs = torch.softmax(outputs, dim=1)
                logits.append(outputs.clone().detach().numpy())
                outputs = outputs.argmax(dim=1)
                predictions.append(outputs.numpy())
                targets.append(y.argmax(dim=1).numpy())
            else:
                outputs = torch.sigmoid(outputs)
                logits.append(outputs.clone().detach().numpy())
                outputs[outputs < conf_thres] = 0
                outputs[outputs >= conf_thres] = 1

                predictions.append(outputs.numpy())
                targets.append(y.numpy())
            total_audioclips += y.shape[0]
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    logits = np.concatenate(logits)

    accuracy = accuracy_score(predictions, targets)

    # Lets do macro for now, as the class distribution is very unbalanced
    f1_per_class = f1_score(predictions, targets, average=None, zero_division=0)
    precision_per_class = precision_score(
        predictions, targets, average=None, zero_division=0
    )
    recall_per_class = recall_score(predictions, targets, average=None, zero_division=0)

    f1 = f1_score(predictions, targets, average="macro", zero_division=0)
    precision = precision_score(predictions, targets, average="macro", zero_division=0)
    recall = recall_score(predictions, targets, average="macro", zero_division=0)
    # marginal_calibration_error_res = marginal_calibration_error(
    #     torch.from_numpy(logits), torch.from_numpy(targets)
    # )

    logs_dict["F1 score"] = f1
    logs_dict["Precision"] = precision
    logs_dict["Recall"] = recall
    logs_dict["Accuracy"] = accuracy
    # logs_dict["Marginal Calibration Error"] = marginal_calibration_error_res

    if eval_run:
        if multi_class:
            # count the number of targets and predictions per class even if they are not present in the batch
            targets_per_class = np.zeros(len(id_to_class_name))
            predictions_per_class = np.zeros(len(id_to_class_name))
            for i in range(len(targets)):
                targets_per_class[targets[i]] += 1
                predictions_per_class[predictions[i]] += 1

            data = {
                "Class": ["all"],
                "# audio clips": [total_audioclips],
                "# targets": [total_audioclips],
                "# predictions": [total_audioclips],
                "Precision": [precision],
                "Recall": [recall],
                "F1-score": [f1],
            }
            for i in range(len(f1_per_class)):
                data["Class"].append(id_to_class_name[i])
                data["# audio clips"].append(targets_per_class[i])
                data["# targets"].append(targets_per_class[i])
                data["# predictions"].append(predictions_per_class[i])
                data["Precision"].append(precision_per_class[i])
                data["Recall"].append(recall_per_class[i])
                data["F1-score"].append(f1_per_class[i])
                logs_dict[
                    "Results Table" if not schreder_data else "Schreder Results Table"
                ] = pd.DataFrame(data)
        else:
            targets_per_class = targets.sum(axis=0)
            predictions_per_class = predictions.sum(axis=0)
            data = {
                "Class": ["all"],
                "# audio clips": [total_audioclips],
                "# targets": [targets.sum()],
                "# predictions": [predictions.sum()],
                "Precision": [precision],
                "Recall": [recall],
                "F1-score": [f1],
            }
            for i in range(len(f1_per_class)):
                data["Class"].append(id_to_class_name[i])
                data["# audio clips"].append(targets_per_class[i])
                data["# targets"].append(targets_per_class[i])
                data["# predictions"].append(predictions_per_class[i])
                data["Precision"].append(precision_per_class[i])
                data["Recall"].append(recall_per_class[i])
                data["F1-score"].append(f1_per_class[i])
                logs_dict[
                    "Results Table" if not schreder_data else "Schreder Results Table"
                ] = pd.DataFrame(data)
    else:
        print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")

    if loss_fn is not None:
        logs_dict["Test loss"] = np.average(losses)
        print(f"Test loss: {np.average(losses)}")

    model.train()

    return logs_dict, (logits, targets)


def create_calibration_plots(
    logits: np.ndarray, targets: np.ndarray, class_names: list[str], save_dir: str
) -> tuple[str, str]:
    reliability_diagram_path = utils.plot_reliability_diagram(
        logits,
        targets,
        logits.shape[-1],
        class_names,
        save_dir,
    )

    # Compute Expected Calibration Error
    ece = utils.compute_expected_calibration_error(
        logits,
        targets,
        logits.shape[-1],
    )
    # Plot Expected Calibration Error per class
    plt.clf()
    fig = plt.figure(figsize=(8, 8), dpi=80)
    sns.set_theme("notebook", style="ticks")
    plt.plot(class_names, ece, "o")
    plt.title("Expected Calibration Error per Class")
    sns.despine()
    ece_path = os.path.join(save_dir, "ece_per_class.png")
    fig.savefig(ece_path)
    return reliability_diagram_path, ece_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)

    args = parser.parse_args()
    params = utils.Params(args.config_path)

    model_name = params.checkpoint.split("/")[-2]
    test_dir = os.path.join(
        os.path.dirname(params.checkpoint),
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(test_dir, exist_ok=True)

    exp_name = model_name + "_" + os.path.basename(params.test_split).rstrip(".pkl")

    wandb.init(
        config=params,
        project="Audio",
        group="rethink",
        job_type="test",
        name=exp_name,
    )

    test_transforms = T.Compose([])

    val_loader = create_dataloader(
        params.test_split,
        params.batch_size,
        params.num_workers,
        spectrogram_transforms=test_transforms,
    )

    model = create_model(
        params.model,
        params.num_classes,
        checkpoint=params.checkpoint,
        device=params.device,
    )

    if params.checkpoint is not None:
        utils.load_checkpoint(params.checkpoint, model)

    id_to_class_name = utils.load_id_to_class_name(params.names)

    logs_dict, (logits, targets) = evaluate(
        model,
        params.device,
        val_loader,
        id_to_class_name,
        eval_run=True,
        conf_thres=params.conf_threshold,
    )

    reliability_diagram_path, ece_path = create_calibration_plots(
        logits, targets, list(id_to_class_name.values()), test_dir
    )
    logs_dict["Reliability Diagram"] = wandb.Image(reliability_diagram_path)
    logs_dict["Expected Calibration Error Plot"] = wandb.Image(ece_path)

    wandb.log(logs_dict)

    wandb.run.finish()
