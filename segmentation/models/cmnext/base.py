import math
import torch
from torch import nn

from .backbones import CMNeXt as CMNeXtBackbone
from .layers import trunc_normal_


def load_dualpath_model(model, model_file):
    # load raw state_dict
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    del state_dict


class BaseModel(nn.Module):
    def __init__(
        self,
        backbone: str = 'CMNeXt-B0',
        num_classes: int = 19,
        modals: list = ['rgb', 'depth', 'event', 'lidar'],
        in_channels=None,
    ):
        super().__init__()
        backbone_name, variant = backbone.split('-')
        assert backbone_name == 'CMNeXt', f"Unsupported backbone: {backbone_name}"
        if in_channels is None:
            in_channels = [3] * len(modals)
        elif isinstance(in_channels, int):
            in_channels = [in_channels]
        else:
            in_channels = list(in_channels)
        self.backbone = CMNeXtBackbone(variant, in_channels=in_channels, modals=modals)
        self.modals = modals

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None):
        if pretrained:
            if len(self.modals) > 1:
                load_dualpath_model(self.backbone, pretrained)
            else:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                if 'model' in checkpoint.keys():
                    checkpoint = checkpoint['model']
                msg = self.backbone.load_state_dict(checkpoint, strict=False)
                print(msg)
