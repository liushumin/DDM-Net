import torch
import torch.nn as nn
import numpy as np
from scipy import signal,misc
import cv2
from torch.nn import functional as F

class DPG(nn.Module):
    def __init__(self, num_channels = 1, base_channels = 24, num_residuals =2):
        super(DPG, self).__init__()

        self.DPG = nn.Sequential(
            nn.Conv2d(num_channels, base_channels, kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(
                    nn.Conv2d(base_channels, base_channels, kernel_size=7, stride=1,
                              padding=3, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(base_channels, base_channels, kernel_size=5, stride=1,
                              padding=2, bias=True),
                    nn.ReLU(inplace=True)) for _ in range(num_residuals)],
            nn.ConvTranspose2d(base_channels, num_channels, kernel_size=5, stride=1,
                               padding=2,
                               bias=True)
            )

    def weight_init(self):
        for m in self._modules:
            weights_init_kaiming(m)

    def forward(self, x):
        conv_filter_M = torch.Tensor(
            [[[[1, 2, 2, 2, 1], [2, 4, 4, 4, 2], [2, 4, 4, 4, 2], [2, 4, 4, 4, 2], [1, 2, 2, 2, 1]]]]).cuda()
        Im_raw = F.conv2d(x, conv_filter_M, padding='same') / 64.0

        x = self.DPG(x)
        x = torch.add(x, Im_raw)
        #torch.clamp(x, min=0, max=1)

        return x


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__=='__main__':
    inputs = np.array([np.float32(a) for a in range(32*60 * 60)]).reshape((32, 1, 60, 60))
    inputs = torch.tensor(inputs)
    net = DPG()
    outputs = net(inputs)
    print(outputs.shape)
