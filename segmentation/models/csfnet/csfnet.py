"""
CSFNet: Cosine Similarity Fusion Network for real-time RGB-X semantic segmentation.

Ported from https://github.com/Danial-Qashqai/CSFNet (STDC backbones + cosine-similarity
fusion gates). The original implementation selects the second-modality channel count and
the adaptive-pooling grid sizes from a hardcoded dataset name (Cityscapes/MFNet/ZJU/FMB).
Here those are exposed as explicit constructor arguments (`extra_in_chans`, `pool_out`) so
the model works with an arbitrary second modality (e.g. PCA-reduced hyperspectral data),
and the decoder always upsamples by scale factor instead of dataset-specific fixed sizes.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def CSFNet(version, num_classes, extra_in_chans, pool_out=None, pretrain=False, backbone_path=None):
    if version == 'CSFNet-1':
        backbone = 'STDCNet813'
    elif version == 'CSFNet-2':
        backbone = 'STDCNet1446'
    else:
        raise ValueError(f'Unknown CSFNet version: {version}')

    return Network(num_class=num_classes, backbone=backbone, extra_in_chans=extra_in_chans,
                    pool_out=pool_out, pretrain=pretrain, backbone_path=backbone_path)


class Network(nn.Module):
    def __init__(self, num_class=19, backbone='STDCNet813', extra_in_chans=3, pool_out=None,
                 pretrain=False, backbone_path=None, act_type='relu'):
        super().__init__()

        if pool_out is None:
            # Adaptive-pooling grid sizes for the fusion attention gates. Tuned for the
            # ~256x256 inputs used in this project; pass an explicit pool_out to override.
            pool_out = [(16, 16), (8, 8), (4, 4), (2, 2), (4, 4)]

        self.backbone_name = backbone
        decoder_channels = [32, 64, 128, 32, num_class]

        self.encoder = Encoder(backbone, extra_in_chans, pretrain, backbone_path, pool_out)
        self.CM = Context_Module(1024, decoder_channels[0], act_type, pooling_size=pool_out[4])
        self.decoder = Decoder(decoder_channels, act_type, pool_out)

    def forward(self, rgb, extra):
        x1, x3, x4, x5 = self.encoder(rgb, extra)
        x5 = self.CM(x5)
        x = self.decoder(x1, x3, x4, x5)

        return x


class Encoder(nn.Module):
    def __init__(self, backbone, extra_in_chans, pretrain, backbone_path, pool_out):
        super().__init__()

        self.backbone_name = backbone
        if backbone == 'STDCNet1446':
            self.encoder = STDCNet1446(extra_in_chans=extra_in_chans, pretrain_model=pretrain,
                                        backbone_path=backbone_path, pool_out=pool_out)
        elif backbone == 'STDCNet813':
            self.encoder = STDCNet813(extra_in_chans=extra_in_chans, pretrain_model=pretrain,
                                       backbone_path=backbone_path, pool_out=pool_out)

    def forward(self, rgb, extra):
        return self.encoder(rgb, extra)


def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 bias=False, act_type='relu', **kwargs):
        if isinstance(kernel_size, (list, tuple)):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        else:
            padding = (kernel_size - 1) // 2 * dilation

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super().__init__()
        activation_hub = {'relu': nn.ReLU, 'relu6': nn.ReLU6,
                           'leakyrelu': nn.LeakyReLU, 'prelu': nn.PReLU,
                           'celu': nn.CELU, 'elu': nn.ELU,
                           'hardswish': nn.Hardswish, 'hardtanh': nn.Hardtanh,
                           'gelu': nn.GELU, 'glu': nn.GLU,
                           'selu': nn.SELU, 'silu': nn.SiLU,
                           'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,
                           'tanh': nn.Tanh, 'none': nn.Identity}

        act_type = act_type.lower()
        if act_type not in activation_hub:
            raise NotImplementedError(f'Unsupported activation type: {act_type}')

        self.activation = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.activation(x)


class decoder_fusion(nn.Module):
    def __init__(self, num_channel, hid_channels, pool_size):
        super().__init__()
        self.attention = fusion_attention(num_channel, hid_channels, pool_size)

    def forward(self, x_high, x_low):
        alpha, beta = self.attention(x_high, x_low)
        return (alpha * x_high) + (beta * x_low)


class Decoder(nn.Module):
    def __init__(self, decoder_channels, act_type, pool_out):
        super().__init__()

        self.CBR1 = ConvBNAct(decoder_channels[0], decoder_channels[0])
        self.CBR2 = ConvBNAct(decoder_channels[0], decoder_channels[1])
        self.CBR3 = ConvBNAct(decoder_channels[1], decoder_channels[2])
        self.CBR4 = ConvBNAct(decoder_channels[2], decoder_channels[3])
        self.CBR5 = ConvBNAct(decoder_channels[3], decoder_channels[4])

        self.D_fusion1 = decoder_fusion(32, 16, pool_out[3])
        self.D_fusion2 = decoder_fusion(64, 32, pool_out[2])
        self.D_fusion3 = decoder_fusion(32, 16, pool_out[0])

    def forward(self, x1, x3, x4, x5):
        x = self.CBR1(x5)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.D_fusion1(x, x4)

        x = self.CBR2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.D_fusion2(x, x3)

        x = self.CBR3(x)
        x = self.CBR4(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.D_fusion3(x, x1)

        x = self.CBR5(x)
        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return out


class Context_Module(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, pooling_size):
        super().__init__()
        hid_channels = in_channels // 4
        self.act_type = act_type

        self.pool1 = self._make_pool_layer(in_channels, hid_channels, pooling_size)
        self.conv2 = self._make_conv_layer(hid_channels, 64, (1, 4), 1)
        self.conv3 = self._make_conv_layer(hid_channels, 64, (4, 1), 1)
        self.conv = conv3x3(64, out_channels)

    def _make_conv_layer(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Activation(act_type='relu')
        )

    def _make_pool_layer(self, in_channels, out_channels, pool_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            ConvBNAct(in_channels, out_channels, 1, act_type=self.act_type)
        )

    def forward(self, x):
        size = x.size()[2:]
        x = self.pool1(x)

        x1 = F.interpolate(self.conv2(x), size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.conv3(x), size, mode='bilinear', align_corners=True)
        return self.conv(x1 + x2)


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, 'block number should be larger than 1.'
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1,
                          groups=out_planes // 2, bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                out = conv(self.avd_layer(out1)) if self.stride == 2 else conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        return torch.cat(out_list, dim=1)


class STDCBackboneBase(nn.Module):
    """
    Shared STDC-Net implementation (two-stream RGB + extra-modality encoder with
    cosine-similarity fusion at three stages). Subclasses set `layers`.
    """
    layers = None

    def __init__(self, extra_in_chans, pool_out, base=64, block_num=4, backbone_path=None,
                 pretrain_model=False):
        super().__init__()
        block = CatBottleneck

        self.backbone_path = backbone_path
        self.extra_in_chans = extra_in_chans

        self.features = self._make_layers(base, self.layers, block_num, block, in_channels=3)
        self.features_d = self._make_layers(base, self.layers, block_num, block, in_channels=extra_in_chans)

        # Stage boundaries within `features`: a 2-layer stem, then one block group per
        # entry in `self.layers` (each group's length is that entry's block count).
        stem_len = 2
        stage0_end = stem_len + self.layers[0]
        stage1_end = stage0_end + self.layers[1]

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[stem_len:stage0_end])
        self.x16 = nn.Sequential(self.features[stage0_end:stage1_end])
        self.x32 = nn.Sequential(self.features[stage1_end:])

        self.x2d = nn.Sequential(self.features_d[:1])
        self.x4d = nn.Sequential(self.features_d[1:2])
        self.x8d = nn.Sequential(self.features_d[stem_len:stage0_end])

        self.f1 = encoder_fusion(32, 16, pool_out[0], skip_connection=True, last_fusion=False)
        self.f2 = encoder_fusion(64, 32, pool_out[1], skip_connection=False, last_fusion=False)
        self.f3 = encoder_fusion(256, 128, pool_out[2], skip_connection=True, last_fusion=True)

        self.agent4 = ConvBNAct(512, 32, 1, act_type='relu')
        self.agent3 = ConvBNAct(256, 64, 1, act_type='relu')
        self.agent1 = ConvBNAct(32, 32, 1, act_type='relu')

        if pretrain_model:
            self.init_weight()

    def init_weight(self):
        pretrain_dict = torch.load(self.backbone_path)['state_dict']
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k not in state_dict:
                continue
            if k.startswith('features.0.conv'):
                model_dict[k] = v
                # Adapt the RGB stem weights to the extra-modality stream by averaging
                # over the input-channel dimension and repeating for `extra_in_chans`.
                mean_weight = torch.mean(v, dim=1, keepdim=True).repeat(1, self.extra_in_chans, 1, 1)
                model_dict[k[:8] + '_d' + k[8:]] = mean_weight
            elif k.startswith('features'):
                model_dict[k] = v
                model_dict[k[:8] + '_d' + k[8:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _make_layers(self, base, layers, block_num, block, in_channels):
        features = [ConvX(in_channels, base // 2, 3, 2), ConvX(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, rgb, extra):
        rgb = self.x2(rgb)
        extra = self.x2d(extra)
        rgb, extra, s1 = self.f1(rgb, extra)
        x1 = self.agent1(s1)

        rgb = self.x4(rgb)
        extra = self.x4d(extra)
        rgb, extra = self.f2(rgb, extra)

        rgb = self.x8(rgb)
        extra = self.x8d(extra)
        s3 = self.f3(rgb, extra)
        x3 = self.agent3(s3)

        s4 = self.x16(s3)
        x4 = self.agent4(s4)

        x5 = self.x32(s4)

        return x1, x3, x4, x5


class STDCNet813(STDCBackboneBase):
    layers = [2, 2, 2]


class STDCNet1446(STDCBackboneBase):
    layers = [4, 5, 3]


class fusion_attention(nn.Module):
    def __init__(self, num_channel, out_channel, pool_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        self.conv1 = nn.Conv2d(num_channel, out_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channel, num_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.activation = nn.Sigmoid()

    def forward(self, rgb, extra):
        rgb = self.pool(rgb).view((rgb.size()[0], rgb.size()[1], -1))
        extra = self.pool(extra).view((extra.size()[0], extra.size()[1], -1))
        similarity_vector = self.cos(rgb, extra).view(rgb.size()[0], rgb.size()[1], 1, 1)

        output = F.relu(self.bn1(self.conv1(similarity_vector)))
        output = self.conv2(output)
        weight = self.activation(output)
        return weight, 1 - weight


class encoder_fusion(nn.Module):
    def __init__(self, num_channel, hid_channels, pool_size, skip_connection, last_fusion):
        super().__init__()
        self.attention = fusion_attention(num_channel, hid_channels, pool_size)
        self.skip_connection = skip_connection
        self.last_fusion = last_fusion

    def forward(self, rgb, extra):
        w_extra, w_rgb = self.attention(rgb, extra)
        w_rgb = rgb.mul(w_rgb)
        w_extra = extra.mul(w_extra)

        if self.skip_connection and not self.last_fusion:
            rgb = rgb + w_extra
            extra = extra + w_rgb
            return rgb, extra, w_rgb + w_extra

        if not self.skip_connection and not self.last_fusion:
            rgb = rgb + w_extra
            extra = extra + w_rgb
            return rgb, extra

        return w_rgb + w_extra
