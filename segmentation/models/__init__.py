import torch

from . import mininet
from . import cmx
from . import csfnet
from . import hyfuseunet
from . import segformer
from . import unet
from . import mmsformer
from . import cmnext


class MiniNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model = mininet.MiniNetv2(in_channels, num_classes, interpolate=True)
        self.num_classes = num_classes

    def forward(self, input):
        return self.model(input)

class MiniNetMultimodal(torch.nn.Module):
    def __init__(self, in_channels: list[int], num_classes: int):
        super().__init__()
        self.model = mininet.MiniNetv2(sum(in_channels), num_classes, interpolate=True)

    def forward(self, inputs):
        input = torch.concat(inputs, axis=1)
        return self.model(input)

class SegFormer(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int, encoder: str):
        super().__init__()
        self.model = segformer.SegFormer(in_chans=in_channels, num_classes=num_classes, backbone=encoder)

    def forward(self, input):
        return self.model(input)

class SegFormerMultimodal(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int, encoder: str):
        super().__init__()
        self.model = segformer.SegFormer(in_chans=sum(in_channels), num_classes=num_classes, backbone=encoder)

    def forward(self, inputs):
        input = torch.concat(inputs, axis=1)
        return self.model(input)

class CMX(torch.nn.Module):
    def __init__(self, in_channels: list[int], num_classes: int, encoder: str):
        super().__init__()
        assert len(in_channels) == 2 and in_channels[0] == 3
        self.model = cmx.EncoderDecoder(extra_in_chans=in_channels[1], num_classes=num_classes, encoder=encoder)

    def forward(self, inputs):
        input1, input2 = inputs
        return self.model(input1, input2)

class CSFNet(torch.nn.Module):
    def __init__(self, in_channels: list[int], num_classes: int, version: str):
        super().__init__()
        assert len(in_channels) == 2 and in_channels[0] == 3
        self.model = csfnet.CSFNet(version, num_classes, extra_in_chans=in_channels[1])

    def forward(self, inputs):
        input1, input2 = inputs
        return self.model(input1, input2)

class MMSFormerMultimodal(torch.nn.Module):
    def __init__(self, in_channels: list[int], num_classes: int, encoder: str):
        super().__init__()
        self.model = mmsformer.MMSFormer(in_channels=in_channels, num_classes=num_classes, backbone=encoder, modals=['img', 'nir'])

    def forward(self, inputs):
        return self.model(inputs)

class CMNeXtMultimodal(torch.nn.Module):
    def __init__(self, in_channels: list[int], num_classes: int, encoder: str):
        super().__init__()
        modals = ['img'] + [f'x{i}' for i in range(1, len(in_channels))]
        self.model = cmnext.CMNeXt(in_channels=in_channels, num_classes=num_classes, backbone=encoder, modals=modals)

    def forward(self, inputs):
        return self.model(inputs)


class HyFuseUNet(torch.nn.Module):
    def __init__(self, in_channels: list[int], num_classes: int):
        super().__init__()
        assert len(in_channels) == 2
        self.model = hyfuseunet.HyFuseUNet(in_channels[0], in_channels[1], num_classes)

    def forward(self, inputs):
        return self.model(inputs)


class UNet(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.model = unet.UNet(in_channels, num_classes)

    def forward(self, input):
        return self.model(input)


class UNetMultimodal(torch.nn.Module):
    def __init__(self, in_channels: list[int], num_classes: int):
        super().__init__()
        self.model = unet.UNet(sum(in_channels), num_classes)

    def forward(self, inputs):
        input = torch.concat(inputs, dim=1)
        return self.model(input)


def create_model(name, in_channels, num_classes):
    if name == 'mininet':
        model = MiniNet(in_channels, num_classes)
    elif name == 'mininet_multimodal':
        model = MiniNetMultimodal(in_channels, num_classes)
    elif name == 'segformer_b0':
        model = SegFormer(in_channels, num_classes, 'mit_b0')
    elif name == 'segformer_b0_multimodal':
        model = SegFormerMultimodal(in_channels, num_classes, 'mit_b0')
    elif name == 'cmx_b0':
        model = CMX(in_channels, num_classes, 'mit_b0')
    elif name == 'csfnet1':
        model = CSFNet(in_channels, num_classes, 'CSFNet-1')
    elif name == 'csfnet2':
        model = CSFNet(in_channels, num_classes, 'CSFNet-2')
    elif name == 'mmsformer_b0':
        model = MMSFormerMultimodal(in_channels, num_classes, 'MMSFormer-B0')
    elif name == 'cmnext_b0':
        model = CMNeXtMultimodal(in_channels, num_classes, 'CMNeXt-B0')
    elif name == 'hyfuseunet':
        model = HyFuseUNet(in_channels, num_classes)
    elif name == 'unet':
        model = UNet(in_channels, num_classes)
    elif name == 'unet_multimodal':
        model = UNetMultimodal(in_channels, num_classes)
    else:
        raise ValueError(f'Unknown model: {name}')
    return model


def create_optimizers(name, model, max_epochs):
    if name == 'mininet':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    elif name == 'mininet_multimodal':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    elif name == 'segformer_b0':
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.1)
    elif name == 'segformer_b0_multimodal':
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.1)
    elif name == 'cmx_b0':
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    elif name == 'csfnet1':
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    elif name == 'csfnet2':
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    elif name == 'mmsformer_b0':
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    elif name == 'cmnext_b0':
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    elif name == 'hyfuseunet':
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    elif name == 'unet':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    elif name == 'unet_multimodal':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    else:
        raise ValueError(f'Unknown model: {name}')

    return optimizer, lr_scheduler
