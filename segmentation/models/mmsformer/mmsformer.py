import torch
from torch.nn import functional as F
from .base import BaseModel
from .heads import SegFormerHead


class MMSFormer(BaseModel):
    def __init__(
        self,
        in_channels,
        num_classes: int,
        backbone: str = 'MMSFormer-B0',
        modals = ['img', 'aolp', 'dolp', 'nir'],
    ):
        super().__init__(backbone, num_classes, modals, in_channels)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: list) -> list:
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x[0].shape[2:], mode='bilinear', align_corners=False)
        return y

    def init_pretrained(self, pretrained: str = None):
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        msg = self.backbone.load_state_dict(checkpoint, strict=False)
        del checkpoint
