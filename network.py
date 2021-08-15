import torch.nn as nn
import torch.nn.functional as F
from unet_parts import *

class LIRNet(nn.Module): 
    def __init__(self):
        super(LIRNet, self).__init__()
        self.n_channels = 3
        self.n_classes = 1
        self.bilinear = False

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128, 3)
        self.down2 = Down(128, 128, 4)
        self.nonlocal2d = NonLocalBlock2D(in_channels = 128,inter_channels = 32,sub_sample = True)
        self.up1 = Up(256, 64, self.bilinear)
        self.up2 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x_nonlocal = self.nonlocal2d(x3)
        x = self.up1(x_nonlocal, x2)
        x = self.up2(x, x1)
        density = F.relu(self.outc(x))
        return density

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)
