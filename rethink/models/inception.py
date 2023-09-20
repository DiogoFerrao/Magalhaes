import torch.nn as nn
import torchvision.models as models


class Inception(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Inception, self).__init__()
        self.model = models.inception_v3(pretrained=pretrained, aux_logits=False)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output
