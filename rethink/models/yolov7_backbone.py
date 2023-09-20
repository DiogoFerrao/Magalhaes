import torch.nn as nn

from yolov7.models.yolo import Model


class Yolov7Backbone(nn.Module):
    def __init__(self, backbone, n_classes):
        super(Yolov7Backbone, self).__init__()
        self.n_classes = n_classes
        self.backbone = Model(backbone)
        self.create_linear_layers(self.n_classes)

    def create_linear_layers(self, n_classes):
        self.linear_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(512, 1024),
            # nn.LazyLinear(1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.linear_layers(x)

        return x
