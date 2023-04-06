import torch
import torch.nn as nn
import numpy as np
from scipy import signal,misc
import cv2
import os
from torch.nn import functional as F

def my_sparased(image,channel):
    spar = np.zeros_like(image)
    hei1 = image.shape[2]
    wid1 = image.shape[3]
    h = channel // 4
    w = channel - h * 4
    spar[:,:,h:hei1 + 1:4, w:wid1 + 1:4] = image[:,:,h:hei1 + 1:4, w:wid1 + 1:4]
    return spar

class DPDN(nn.Module):
    def __init__(self,DPG_num_channels =1, DPG_base_channels = 24,DPG_num_residuals = 2,DDM_num_channels = 2, DDM_base_channels = 64, DDM_num_residuals =3):
        super(DPDN, self).__init__()

        self.DPG = nn.Sequential(nn.Conv2d(DPG_num_channels, DPG_base_channels, kernel_size=9, stride=1, padding=4, bias=True),
                                 nn.ReLU(inplace=True),
                                 *[
                                     nn.Sequential(
                                         nn.Conv2d(DPG_base_channels, DPG_base_channels, kernel_size=7, stride=1,
                                                   padding=3, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(DPG_base_channels, DPG_base_channels, kernel_size=5, stride=1,
                                                   padding=2, bias=True),
                                         nn.ReLU(inplace=True)) for _ in range(DPG_num_residuals)],
                                 nn.ConvTranspose2d(DPG_base_channels, DPG_num_channels, kernel_size=5, stride=1,
                                                    padding=2,
                                                    bias=True)
                                 )

        self.input_conv = nn.Sequential(nn.Conv2d(129, DDM_base_channels, kernel_size=3, stride=1, padding=1, bias=True),nn.ReLU(inplace=True))
        self.residual_layers = nn.Sequential(*[nn.Sequential(nn.Conv2d(DDM_base_channels, DDM_base_channels, kernel_size=3, stride=1, padding=1, bias=False),nn.ReLU(inplace=True)) for _ in range(DDM_num_residuals)])
        self.output_conv = nn.ConvTranspose2d(DDM_base_channels, DDM_num_channels//2, kernel_size=3, stride=1, padding=1, bias=True)
        self.layers = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=DDM_base_channels, kernel_size=9, stride=1, padding=4,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=DDM_base_channels, out_channels=DDM_base_channels // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=DDM_base_channels // 2, out_channels=1 * (1 ** 2), kernel_size=5,
                      stride=1, padding=2, bias=True),
            nn.PixelShuffle(1)
        )  #SRCNN

        self.extract = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,
                      stride=1, padding=1, bias=True))  # before error FENET

        self.extract2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,
                      stride=1, padding=1, bias=True))  # before error FENET

    def weight_init(self):
        for m in self._modules:
            weights_init_kaiming(m)

    def forward(self, mosaicked, sparse_image):

        conv_filter_M = torch.Tensor(
            [[[[1, 2, 2, 2, 1], [2, 4, 4, 4, 2], [2, 4, 4, 4, 2], [2, 4, 4, 4, 2], [1, 2, 2, 2, 1]]]]).cuda()
        conv_filter_H = torch.Tensor(
            [[[[1, 2, 3, 4, 3, 2, 1], [2, 4, 6, 8, 6, 4, 2], [3, 6, 9, 12, 9, 6, 3], [4, 8, 12, 16, 12, 8, 4],
               [3, 6, 9, 12, 9, 6, 3], [2, 4, 6, 8, 6, 4, 2], [1, 2, 3, 4, 3, 2, 1]]]]).cuda()

        Im_raw = F.conv2d(mosaicked, conv_filter_M, padding='same') / 64.0

        residual_info = self.DPG(mosaicked)

        PPI_estimated = torch.add(residual_info, Im_raw)
        #torch.clamp(PPI_estimated, min=0, max=1)

        Demosaic = F.conv2d(sparse_image, conv_filter_H, padding='same') / 16.0

        Demosaic_feature = self.extract2(torch.cat((Demosaic,sparse_image),1))
        PPI_feature = self.extract(PPI_estimated)

        input = torch.cat((Demosaic_feature,PPI_feature,sparse_image),dim=1)
        demosaic_residual = self.input_conv(input)
        demosaic_residual = self.residual_layers(demosaic_residual)
        demosaic_residual = self.output_conv(demosaic_residual)

        demosaic_estimated = torch.add(demosaic_residual, Demosaic)
        #torch.clamp(demosaic_estimated, min=0, max=1)

        return PPI_estimated,demosaic_estimated


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
    mosaicked = np.array([np.float32(a) for a in range(32*60 * 60)]).reshape((32, 1, 60, 60))
    PPI = np.array([np.float32(a) for a in range(32*60 * 60)]).reshape((32, 1, 60, 60))
    mosaicked = torch.tensor(mosaicked)
    PPI = torch.tensor(PPI)
    net = FENET()
    outputs = net(mosaicked,PPI)
    print(outputs.shape)
