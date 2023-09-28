import json
import math
import os
import shutil

import matplotlib.pyplot as plt
import ml_insights as mli
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.beta import Beta


class Params:
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            params = json.dump(self.__dict__, f, indent=4)  # noqa: F841

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


class RunningAverage:
    def __init__(self):
        self.total = 0
        self.steps = 0

    def update(self, loss):
        self.total += loss
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def save_checkpoint(state, is_best, split, checkpoint):
    filename = os.path.join(checkpoint, "last{}.pth".format(split))
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist")
        os.mkdir(checkpoint)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename, os.path.join(checkpoint, "model_best_{}.pth".format(split))
        )


def load_best_model(params, split, run_name, model):
    best_model_path = os.path.join(
        params.checkpoint_dir, run_name, f"model_best_{split}.pth"
    )
    if os.path.exists(best_model_path):
        model.load_state_dict(
            torch.load(best_model_path, map_location=torch.device("cpu"))["model"]
        )


def load_checkpoint(checkpoint, model, optimizer=None, parallel=False):
    if not os.path.exists(checkpoint):
        raise ("File Not Found Error {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location="cpu")
    if parallel:
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def initialize_weights(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find("Linear") != -1:
        nn.init.ones_(m.weight.data)


def load_id_to_class_name(path):
    id_to_class = {}
    for i, name in enumerate(open(path).readlines()):
        if name.strip() == "":
            continue
        id_to_class[i] = name.strip()

    return id_to_class


def exp_warmup_linear_down(warmup, rampdown_length, start_rampdown, last_value):
    rampup = exp_rampup(warmup)
    rampdown = linear_rampdown(rampdown_length, start_rampdown, last_value)

    def wrapper(epoch):
        return rampup(epoch) * rampdown(epoch)

    return wrapper


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""

    def wrapper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.5, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0

    return wrapper


def linear_rampdown(rampdown_length, start=0, last_value=0):
    def wrapper(epoch):
        if epoch <= start:
            return 1.0
        elif epoch - start < rampdown_length:
            return (
                    last_value
                    + (1.0 - last_value)
                    * (rampdown_length - epoch + start)
                    / rampdown_length
            )
        else:
            return last_value

    return wrapper


def mixup(size, alpha):
    rn_indices = torch.randperm(size)
    lambd = np.random.beta(alpha, alpha, size).astype(np.float32)
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    lam = torch.FloatTensor(lambd)
    return rn_indices, lam


def mixstyle(x, p=0.4, alpha=0.4, eps=1e-6, mix_labels=False):
    if np.random.rand() > p:
        return x
    batch_size = x.size(0)

    # changed from dim=[2,3] to dim=[1,3] - from channel-wise statistics to frequency-wise statistics
    f_mu = x.mean(dim=[1, 3], keepdim=True)
    f_var = x.var(dim=[1, 3], keepdim=True)

    f_sig = (f_var + eps).sqrt()  # compute instance standard deviation
    f_mu, f_sig = f_mu.detach(), f_sig.detach()  # block gradients
    x_normed = (x - f_mu) / f_sig  # normalize input
    lmda = (
        Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device)
    )  # sample instance-wise convex weights
    perm = torch.randperm(batch_size).to(x.device)  # generate shuffling indices
    f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]  # shuffling
    mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)  # generate mixed mean
    sig_mix = f_sig * lmda + f_sig_perm * (
            1 - lmda
    )  # generate mixed standard deviation
    x = x_normed * sig_mix + mu_mix  # denormalize input using the mixed statistics
    if mix_labels:
        return x, perm, lmda
    return x


def plot_reliability_diagram(
        y_hat: np.ndarray,
        y_target: np.ndarray,
        num_classes: int,
        class_names: list[str],
        save_dir: str,
) -> str:
    assert len(class_names) == num_classes
    num_rows = math.ceil(num_classes / 3)
    fig = plt.figure(figsize=(6.6 * num_rows, 20))
    for i in range(num_classes):
        fig.add_subplot(num_rows, 3, i + 1)
        mli.plot_reliability_diagram(
            y_target[:, i],
            y_hat[:, i],
            reliability_title=class_names[i],
            plot_style="seaborn-notebook",
        )
    fig_path = os.path.join(save_dir, "reliability_diagram.png")
    fig.savefig(fig_path)
    return fig_path


def compute_expected_calibration_error(
        y_hat: np.ndarray,
        y_target: np.ndarray,
        num_classes: int,
        n_bins=20,
) -> list[float]:
    """This function computes the expected calibration error for each class.
    Based on: https://medium.com/@wolframalphav1.0/evaluate-the-performance-of-a-model-in-high-risk-applications-using-expected-calibration-error-and-dbc392c68318

    Args:
        y_hat (np.ndarray): The predicted probabilities for each class.
        y_target (np.ndarray): The target probabilities for each class.
        num_classes (int): The number of classes.
        class_names (list[str]): The names of the classes.

    Returns:
        np.ndarray: The expected calibration error for each class.
    """
    expected_caliberation_error_arr = []
    # overconfidence_error_arr = []

    for class_id in range(num_classes):
        y_prob = y_hat[:, class_id].reshape(-1, 1)
        y_true = y_target[:, class_id].reshape(-1, 1)

        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)

        y_prob_max = np.max(y_prob, axis=-1)
        binids = np.digitize(y_prob_max, bins) - 1

        y_correct_classified = (
                np.argmax(y_true, axis=-1) == np.argmax(y_prob, axis=-1)
        ).astype(int)

        bin_sums = np.bincount(binids, weights=y_prob_max, minlength=len(bins))
        bin_true = np.bincount(
            binids, weights=y_correct_classified, minlength=len(bins)
        )
        bin_total = np.bincount(binids, minlength=len(bins))

        nonzero = bin_total != 0

        # acc(Bm)
        prob_true = bin_true[nonzero] / bin_total[nonzero]

        # conf(Bm)
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]

        expected_caliberation_error = (
                np.sum(bin_total[nonzero] * np.abs(prob_true - prob_pred))
                / bin_total[nonzero].sum()
        )
        # overconfidence_error = np.sum(
        #     bin_total[nonzero]
        #     * prob_pred
        #     * np.max(
        #         np.concatenate(
        #             (
        #                 (prob_pred - prob_true).reshape(-1, 1),
        #                 np.zeros((1, len(prob_pred))).T,
        #             ),
        #             axis=1,
        #         ),
        #         axis=-1,
        #     )
        #     / bin_total[nonzero].sum()
        # )
        expected_caliberation_error_arr.append(expected_caliberation_error)
        # overconfidence_error_arr.append(overconfidence_error)
    return expected_caliberation_error_arr
