import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class _BasicBlock(nn.Module):
    # no_relu=True defers activation for bilateral fusion
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, no_relu: bool = False):
        super().__init__()
        self.conv1   = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2     = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.relu    = nn.ReLU(inplace=True)
        self.no_relu = no_relu
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + self.shortcut(x)
        return out if self.no_relu else self.relu(out)


def _make_layer(in_ch: int, out_ch: int, n: int,
                stride: int = 1, fuse_last: bool = False) -> nn.Sequential:
    blocks = []
    for i in range(n):
        blocks.append(_BasicBlock(
            in_ch if i == 0 else out_ch,
            out_ch,
            stride=stride if i == 0 else 1,
            no_relu=(i == n - 1 and fuse_last),
        ))
    return nn.Sequential(*blocks)


def _compress(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.BatchNorm2d(in_ch, momentum=0.01, eps=1e-3),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
    )


def _down_bilateral(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.BatchNorm2d(in_ch, momentum=0.01, eps=1e-3),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
    )


class _DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module (Pan et al., 2022)."""

    def __init__(self, in_ch: int, branch_ch: int, out_ch: int):
        super().__init__()

        def _pool(pool: nn.Module) -> nn.Sequential:
            return nn.Sequential(
                pool,
                nn.BatchNorm2d(in_ch,     momentum=0.01, eps=1e-3),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            )

        def _proc() -> nn.Sequential:
            return nn.Sequential(
                nn.BatchNorm2d(branch_ch, momentum=0.01, eps=1e-3),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_ch, branch_ch, 3, padding=1, bias=False),
            )

        self.s0 = nn.Sequential(
            nn.BatchNorm2d(in_ch, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
        )
        self.s1 = _pool(nn.AvgPool2d(5,  stride=2, padding=2))
        self.s2 = _pool(nn.AvgPool2d(9,  stride=4, padding=4))
        self.s3 = _pool(nn.AvgPool2d(17, stride=8, padding=8))
        self.s4 = _pool(nn.AdaptiveAvgPool2d(1))
        self.p1, self.p2, self.p3, self.p4 = _proc(), _proc(), _proc(), _proc()
        self.compress = nn.Sequential(
            nn.BatchNorm2d(branch_ch * 5, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_ch * 5, out_ch, 1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_ch, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        up   = lambda t: F.interpolate(t, size=(h, w), mode='bilinear', align_corners=True)
        x0 = self.s0(x)
        x1 = self.p1(up(self.s1(x)) + x0)
        x2 = self.p2(up(self.s2(x)) + x1)
        x3 = self.p3(up(self.s3(x)) + x2)
        x4 = self.p4(up(self.s4(x)) + x3)
        return self.compress(torch.cat([x0, x1, x2, x3, x4], dim=1)) + self.shortcut(x)


class DDRNet23Slim(nn.Module):
    """DDRNet-23-slim (C=32) from Pan et al., 2022.

    Returns (main, aux4, aux3) during training with deep_supervision=True, else a plain tensor.
    """

    def __init__(self, num_classes: int = 1, deep_supervision: bool = True):
        super().__init__()
        C = 32
        self.deep_supervision = deep_supervision

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )

        self.layer1  = _make_layer(C,   C,   2, stride=1)
        self.layer2  = _make_layer(C,   C*2, 2, stride=2)

        self.layer3_ = _make_layer(C*2, C*2, 2, stride=1, fuse_last=True)
        self.layer3  = _make_layer(C*2, C*4, 2, stride=2, fuse_last=True)
        self.comp3   = _compress(C*4, C*2)
        self.down3   = _down_bilateral(C*2, C*4)

        self.layer4_ = _make_layer(C*2, C*2, 2, stride=1, fuse_last=True)
        self.layer4  = _make_layer(C*4, C*8, 2, stride=2, fuse_last=True)
        self.comp4   = _compress(C*8, C*2)
        self.down4   = nn.Sequential(
            nn.BatchNorm2d(C*2, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(C*2, C*4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C*4, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(C*4, C*8, 3, stride=2, padding=1, bias=False),
        )

        self.layer5_ = _make_layer(C*2, C*2, 1, stride=1)
        self.layer5  = _make_layer(C*8, C*8, 1, stride=1)

        self.spp  = _DAPPM(C*8, C*2, C*4)

        self.head = nn.Sequential(
            nn.BatchNorm2d(C*2 + C*4, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(C*2 + C*4, C*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C*2, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(C*2, num_classes, 1),
        )

        if deep_supervision:
            self.aux3 = nn.Conv2d(C*2, num_classes, 1)
            self.aux4 = nn.Conv2d(C*2, num_classes, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        H, W   = x.shape[2:]
        h8, w8 = H // 8, W // 8

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x_h = self.layer3_(x)
        x_l = self.layer3(x)
        x_l = x_l + self.down3(x_h)
        x_h = x_h + F.interpolate(self.comp3(x_l), size=(h8, w8),
                                   mode='bilinear', align_corners=False)
        x_l = F.relu(x_l)
        x_h = F.relu(x_h)
        aux3_feat = x_h

        x_h = self.layer4_(x_h)
        x_l = self.layer4(x_l)
        x_l = x_l + self.down4(x_h)
        x_h = x_h + F.interpolate(self.comp4(x_l), size=(h8, w8),
                                   mode='bilinear', align_corners=False)
        x_l = F.relu(x_l)
        x_h = F.relu(x_h)
        aux4_feat = x_h

        x_h = self.layer5_(x_h)
        x_l = F.interpolate(self.spp(self.layer5(x_l)), size=(h8, w8),
                             mode='bilinear', align_corners=False)

        out = F.interpolate(self.head(torch.cat([x_h, x_l], dim=1)),
                            size=(H, W), mode='bilinear', align_corners=False)

        if self.training and self.deep_supervision:
            aux3 = F.interpolate(self.aux3(aux3_feat), size=(H, W),
                                 mode='bilinear', align_corners=False)
            aux4 = F.interpolate(self.aux4(aux4_feat), size=(H, W),
                                 mode='bilinear', align_corners=False)
            return out, aux4, aux3

        return out
