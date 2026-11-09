"""LineNet: small two-head segmentation (line + mat), same output idea as TwinLiteNet."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DS(nn.Module):
    """Depthwise-separable conv with a residual connection when shapes allow."""

    def __init__(self, c_in, c_out, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, stride, dilation, dilation, groups=c_in, bias=False),
            nn.BatchNorm2d(c_in),
            nn.ReLU6(inplace=True),
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.res = stride == 1 and c_in == c_out
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        y = self.net(x)
        return self.act(x + y if self.res else y)


class LineNet(nn.Module):
    """Input (N,3,120,320) BGR in [-1,1]. Output logits (N,2,120,320): 0=line, 1=mat."""

    def __init__(self, base=16):
        super().__init__()
        c1, c2, c3 = base, base * 3 // 2, base * 3
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, 2, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU6(inplace=True))
        self.down2 = nn.Sequential(DS(c1, c2, 2), DS(c2, c2))
        # Dilation gives the receptive field needed to follow curved, distant lines.
        self.down3 = nn.Sequential(DS(c2, c3, 2), DS(c3, c3, dilation=2), DS(c3, c3, dilation=4))
        self.up2 = DS(c3 + c2, c2)
        self.up1 = DS(c2 + c1, c1)
        self.head = nn.Conv2d(c1, 2, 1)

    def forward(self, x):
        size = x.shape[-2:]
        s1 = self.stem(x)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        y = F.interpolate(s3, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up2(torch.cat([y, s2], 1))
        y = F.interpolate(y, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up1(torch.cat([y, s1], 1))
        y = self.head(y)
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)
