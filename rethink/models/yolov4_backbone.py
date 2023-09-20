import torch.nn as nn


class Yolov4Backbone(nn.Module):
    def __init__(self, module_list, routs, n_classes):
        super(Yolov4Backbone, self).__init__()
        self.module_list = module_list
        # self.linear_layers = nn.Sequential(
        #     nn.AvgPool2d(3),
        #     nn.Flatten(),
        #     # nn.LayerNorm(2048),
        #     # nn.ReLU(),
        #     # nn.Dropout(p=0.2),
        #     nn.LazyLinear(n_classes),
        # )
        self.linear_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.LazyLinear(1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, n_classes),
        )
        self.routs = routs

    def forward(self, x):
        out = []
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in [
                "WeightedFeatureFusion",
                "FeatureConcat",
                "FeatureConcat2",
                "FeatureConcat3",
                "FeatureConcat_l",
                "ScaleChannel",
                "ScaleSpatial",
            ]:
                x = module(x, out)
            else:
                x = module(x)
            out.append(x if self.routs[i] else [])

        x = self.linear_layers(x)

        return x
