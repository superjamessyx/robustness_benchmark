import torch
import torch.nn as nn
import torchvision.models as models
from modeling import build_head, build_backbone

class CamelyonClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.mobilenet_v2(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.pool = nn.AvgPool2d(3, 1)
        self.fc = nn.Linear(1280, 2)

        n_params = sum([p.numel() for p in self.parameters()])

        print("\n")
        print("# " * 50)
        print("MobileNet v2 initialized with {:.3e} parameters".format(n_params))
        print("# " * 50)
        print("\n")

    def forward(self, x):

        return self.fc(self.pool(self.backbone(x)).view(x.shape[0], -1))

    def print_modules(self):
        for idx, param in enumerate(self.modules()):
            print("Module : ", idx)
            print(param)
            print("\n")


class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, model_name, num_classes, dropout=0.0, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_name,
            pretrained=True,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.classifier = nn.Linear(fdim, num_classes)
        self.drop = nn.Dropout(dropout)
        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        f = self.drop(f)
        y = self.classifier(f)

        if return_feature:
            return y, f

        return y

if __name__ == '__main__':

    zeros = torch.zeros((2, 3, 96, 96))
    model = CamelyonClassifier()
    print(model(zeros).shape)