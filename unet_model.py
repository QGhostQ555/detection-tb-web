import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(F.max_pool2d(x, 2))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))

class LungUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c*2)
        self.down2 = Down(c*2, c*4)
        self.down3 = Down(c*4, c*8)
        self.down4 = Down(c*8, c*16)
        self.up1 = Up(c*16 + c*8, c*8)
        self.up2 = Up(c*8 + c*4, c*4)
        self.up3 = Up(c*4 + c*2, c*2)
        self.up4 = Up(c*2 + c, c)
        self.outc = nn.Conv2d(c, out_channels, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5,x4); x = self.up2(x,x3); x = self.up3(x,x2); x = self.up4(x,x1)
        return self.outc(x)

class AttentionGate(nn.Module):
    def __init__(self, gate_c, skip_c, inter_c):
        super().__init__()
        self.gate_proj = nn.Sequential(nn.Conv2d(gate_c, inter_c, 1), nn.BatchNorm2d(inter_c))
        self.skip_proj = nn.Sequential(nn.Conv2d(skip_c, inter_c, 1), nn.BatchNorm2d(inter_c))
        self.psi = nn.Sequential(nn.Conv2d(inter_c, 1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, gate, skip):
        gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        att = self.psi(self.relu(self.gate_proj(gate) + self.skip_proj(skip)))
        return skip * att

class AttentionUp(nn.Module):
    def __init__(self, gate_c, skip_c, out_c):
        super().__init__()
        self.attention = AttentionGate(gate_c, skip_c, max(1, skip_c//2))
        self.conv = DoubleConv(gate_c + skip_c, out_c)
    def forward(self, x, skip):
        skip = self.attention(x, skip)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))

class AttentionLungUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c*2)
        self.down2 = Down(c*2, c*4)
        self.down3 = Down(c*4, c*8)
        self.down4 = Down(c*8, c*16)
        self.up1 = AttentionUp(c*16, c*8, c*8)
        self.up2 = AttentionUp(c*8, c*4, c*4)
        self.up3 = AttentionUp(c*4, c*2, c*2)
        self.up4 = AttentionUp(c*2, c, c)
        self.outc = nn.Conv2d(c, out_channels, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5,x4); x = self.up2(x,x3); x = self.up3(x,x2); x = self.up4(x,x1)
        return self.outc(x)