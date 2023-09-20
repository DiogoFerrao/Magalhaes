import os
from copy import deepcopy
from typing import Optional, Union

import torch
import torch.nn as nn

import rethink.models.yolov4_backbone
import rethink.models.yolov7_tiny_backbone
import rethink.models.yolov7_backbone
from rethink.passt.model import get_model

from yolo.models.models import create_modules
from yolo.utils.parse_config import parse_model_cfg


def create_yolo_backbone(
    num_classes: int,
    checkpoint: Optional[str] = None,
):
    cfg_path = os.path.join(
        os.path.dirname(__file__), "../../yolo/models/yolov4-csp-backbone.cfg"
    )
    backbone_module_list, routs = create_modules(
        parse_model_cfg(cfg_path), (0, 0), cfg_path
    )
    model = rethink.models.yolov4_backbone.Yolov4Backbone(
        backbone_module_list, routs, num_classes
    )
    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)

    return model


def create_pretrained_yolov7_backbone(
    num_classes: int,
    cfg: str,
    weights_path: Optional[str] = None,
):
    """Create a pretrained yolov7 backbone.

    Args:
        num_classes (int): Number of classes.
        cfg (str): Path to the config file.
        weights_path (Optional[str], optional): Path to the weights. Defaults to None.

    Returns:
        Yolov7Backbone: Yolov7 backbone.
    """
    model = rethink.models.yolov7_backbone.Yolov7Backbone(
        os.path.join(os.path.dirname(__file__), cfg), num_classes
    )
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu")
        # some checkpoint have a different keys
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict.keys():
            state_dict = state_dict["model"]
        else:
            raise ValueError("Unknown checkpoint format")

        if not isinstance(state_dict, dict):
            state_dict = state_dict.state_dict()

        if list(state_dict.keys())[0].split(".")[0] != "backbone":
            for k in list(state_dict.keys()):
                state_dict[f"backbone.{k}"] = state_dict[k]
                del state_dict[k]

        if list(state_dict.keys())[0].split(".")[1] != "model":
            for k in list(state_dict.keys()):
                k_arr = k.split(".")
                k_arr = k_arr[:1] + ["model"] + k_arr[1:]
                state_dict[".".join(k_arr)] = state_dict[k]
                del state_dict[k]

        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        print("Loaded pretrained model with incompatible keys: ", incompatible_keys)
    return model


def create_distilled_yolov7_backbone(
    num_classes: int, cfg: str, weights_path: Optional[str]
):
    if weights_path is None:
        raise ValueError("Path to model weights is None")

    # Audioset has 527 classes
    model = rethink.models.yolov7_backbone.Yolov7Backbone(
        os.path.join(os.path.dirname(__file__), cfg), 527
    )
    state_dict = torch.load(weights_path, map_location="cpu")["model"]
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    print("Loaded pretrained model with incompatible keys: ", incompatible_keys)
    model.create_linear_layers(num_classes)
    return model


def create_passt_backbone(num_classes):
    arch = "passt_s_swa_p16_128_ap476"
    fstride = 10
    tstride = 10
    model = get_model(arch=arch, fstride=fstride, tstride=tstride)
    model.head = nn.Sequential(
        nn.LayerNorm(model.num_features), nn.Linear(model.num_features, num_classes)
    )
    return model


def create_model(
    arch: str,
    num_classes: int,
    checkpoint: Optional[str] = None,
    device: Union[str, torch.device] = "cuda:0",
):
    if checkpoint == "":
        checkpoint = None
    model = None
    if arch == "yolo":
        model = create_yolo_backbone(num_classes, checkpoint)
    elif "yolov7" in arch:
        CONFIGS = {
            "yolov7": "yolov7-backbone.yaml",
            "yolov7_distilled": "yolov7-backbone.yaml",
            "yolov7_tiny": "yolov7-tiny-backbone.yaml",
            "yolov7_tiny_distilled": "yolov7-tiny-backbone.yaml",
        }
        cfg_path = os.path.join(os.path.dirname(__file__), CONFIGS[arch])
        if "distilled" in arch:
            model = create_distilled_yolov7_backbone(num_classes, cfg_path, checkpoint)
        else:
            model = create_pretrained_yolov7_backbone(num_classes, cfg_path, checkpoint)

    elif arch == "passt":
        model = create_passt_backbone(num_classes)
    elif arch == "cnn":
        model = nn.Sequential(
            nn.Conv2d(3, 8, 6),
            nn.AvgPool2d(2, 1),
            nn.Conv2d(8, 16, 3),
            nn.AvgPool2d(2, 1),
            nn.Flatten(),
            nn.LazyLinear(124),
            nn.Linear(124, num_classes),
        )
    else:
        print("Unknown architecture")
        exit(0)
    return model.to(device)


class ModelEma(nn.Module):
    """Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
