import torch.nn as nn
import timm


class Res2NetBackbone(nn.Module):
    def __init__(self, model_name='res2net50_26w_4s', pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

    def forward(self, x):
        features = self.backbone(x)
        return tuple(features)


def res2net50_26w_4s(pretrained=False):
    return Res2NetBackbone(model_name='res2net50_26w_4s', pretrained=pretrained)
