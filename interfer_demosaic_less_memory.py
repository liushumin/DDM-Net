###\u6709\u5750\u6807\u56fe\u8f93\u5165\u7684\u5355\u6a21\u578b\u591a\u56fe\u8bc4\u4ef7\u65b9\u6cd5
import argparse
import cv2
import math
import numpy as np
import os
import torch
import time
from torch.autograd import Variable
from libtiff import TIFF, TIFFfile, TIFFimage
from networks.DPDN import DPDN
from PIL import Image
from scipy import signal
from My_function import reorder_imec, reorder_2filter


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return signal.convolve2d(x, np.rot90(kernel, 2), mode=mode)

def mask_input(GT_image):
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], 16), dtype=np.float32)
    mask[0::4, 0::4, 0] = 1
    mask[0::4, 1::4, 1] = 1
    mask[0::4, 2::4, 2] = 1
    mask[0::4, 3::4, 3] = 1
    mask[1::4, 0::4, 4] = 1
    mask[1::4, 1::4, 5] = 1
    mask[1::4, 2::4, 6] = 1
    mask[1::4, 3::4, 7] = 1
    mask[2::4, 0::4, 8] = 1
    mask[2::4, 1::4, 9] = 1
    mask[2::4, 2::4, 10] = 1
    mask[2::4, 3::4, 11] = 1
    mask[3::4, 0::4, 12] = 1
    mask[3::4, 1::4, 13] = 1
    mask[3::4, 2::4, 14] = 1
    mask[3::4, 3::4, 15] = 1
    input_image = mask * GT_image
    return input_image

def compute_PSNR(estimated,real):
    estimated = np.float64(estimated)
    real = np.float64(real)
    MSE = np.mean((estimated-real)**2)
    PSNR = 10*np.log10(255*255/MSE)
    return PSNR

def compute_ssim_channel(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))

def compute_ssim(estimate,real):
    SSIM_totoal  =0
    for i in range(estimate.shape[2]):
        SSIM = compute_ssim_channel(estimate[:, :, i], real[:, :, i])
        SSIM_totoal += SSIM
    return SSIM_totoal / 16.0

def compute_sam(x_true,x_pre):
    buff1 = x_true*x_pre
    buff2 = np.sum(buff1, 2)
    buff2[buff2 == 0] = 2.2204e-16
    buff4 = np.sqrt(np.sum(x_true * x_true, 2))
    buff4[buff4 == 0] = 2.2204e-16
    buff5 = np.sqrt(np.sum(x_pre * x_pre, 2))
    buff5[buff5 == 0] = 2.2204e-16
    buff6 = buff2/buff4
    buff8 = buff6/buff5
    buff8[buff8 > 1] = 1
    buff10 = np.arccos(buff8)
    buff9 = np.mean(np.arccos(buff8))
    SAM = (buff9) * 180 / np.pi
    return SAM


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description="PyTorch LapSRN Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoint/model_in_paper.pth", type=str, help="model path")
#parser.add_argument("--model", default="checkpoint/fine-tuning/final_model_epoch_4000.pth", type=str, help="model path")
parser.add_argument("--dataset", default="test_data", type=str, help="dataset name, Default: CAVE")
parser.add_argument("--scale", default=4, type=int, help="msfa_size, Default: 4")

opt = parser.parse_args()
cuda = True
norm_flag = False

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print(opt.model)

model= DPDN()
m_state_dict = torch.load(opt.model)
model.load_state_dict(m_state_dict['model'].state_dict())

image_list = opt.dataset
avg_psnr_predicted = 0.0
avg_psnr_PPID = 0.0
avg_sam_predicted = 0.0
avg_sam_PPID = 0.0
avg_ssim_predicted_MCAN = 0.0
avg_ssim_predicted_our= 0.0
avg_ssim_PPID = 0.0
avg_ergas_predicted = 0.0
avg_elapsed_time = 0.0
avg_PPI_psnr_predicted = 0.0
sample_num = 0
knum = 1
with torch.no_grad():
    for ijk in range(knum):
        for i in range(1):
            for image_name in sorted(os.listdir(image_list)):
                print("Processing", image_name)
                sample_num = sample_num + 1
                im_gt_y = np.load(opt.dataset + "/" + image_name)  # 512*512*16
                ###\u539f\u672c\u7684im_gt_y\u6309\u7167\u5b9e\u9645\u76f8\u673a\u6ee4\u6ce2\u9635\u5217\u6392\u5217
                im_l_y = mask_input(im_gt_y)
                ###\u6309\u7167\u5b9e\u9645\u76f8\u673a\u6ee4\u6ce2\u9635\u5217\u6392\u5217\u9006\u8fd8\u539f\u4e3a\u4ece\u5927\u5230\u5c0f\u7684\u987a\u5e8f
                im_l_y = reorder_imec(im_l_y)
                im_gt_y = reorder_imec(im_gt_y)
                im_input = im_l_y



                im_gt_y = im_gt_y.transpose(2, 0, 1)
                im_l_y = im_l_y.transpose(2, 0, 1)
                im_input = im_input.transpose(2, 0, 1) # C H W
                raw = im_input.sum(axis=0)   # MSFA

                im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[1], im_input.shape[2])
                raw = Variable(torch.from_numpy(raw).float()).view(1, -1, raw.shape[0], raw.shape[1])
                estimated_Demosaic = np.zeros_like(im_l_y)
                for index in range(16):
                    sparse_image = im_l_y[index,:,:]
                    sparse_image = Variable(torch.from_numpy(sparse_image).float()).view(1, -1, sparse_image.shape[0], sparse_image.shape[1])
                    if cuda:
                        # PPI_net = PPI_net.cuda()
                        model = model.cuda()

                        im_input = im_input.cuda()
                        raw = raw.cuda()
                        sparse_image = sparse_image.cuda()
                    else:
                        # PPI_net  =  PPI_net.cpu()
                        model = model.cpu()

                    # estimated_PPI = PPI_net(raw)
                    estimated_PPI, estimated_demosaic = model(raw, sparse_image)
                    estimated_demosaic = estimated_demosaic.cpu()
                    estimated_demosaic = estimated_demosaic.data[0].numpy().astype(np.float32)
                    estimated_demosaic = estimated_demosaic * 255.
                    estimated_demosaic[estimated_demosaic < 0] = 0
                    estimated_demosaic[estimated_demosaic > 255.] = 255.

                    estimated_Demosaic[index,:,:] = estimated_demosaic

                target_PPI = im_gt_y.sum(axis=0) / 16.
                target_PPI = Variable(torch.from_numpy(target_PPI).float()).view(1, -1, target_PPI.shape[0], target_PPI.shape[1])

                target_PPI = target_PPI.cuda()
                target_PPI = target_PPI

                estimated_PPI = estimated_PPI.cpu()
                estimated_PPI = estimated_PPI.data[0].numpy().astype(np.float32)

                estimated_PPI = estimated_PPI * 255.
                estimated_PPI[estimated_PPI < 0] = 0
                estimated_PPI[estimated_PPI > 255.] = 255.

                raw = raw.cpu()
                raw = raw.data[0].numpy().astype(np.float32)
                raw = raw * 255.
                raw[raw < 0] = 0
                raw[raw > 255.] = 255.

                im_input = im_input.cpu()
                im_input = im_input.data[0].numpy().astype(np.float32)
                im_input = im_input * 255.
                im_input[im_input < 0] = 0
                im_input[im_input > 255.] = 255.

                target_PPI = target_PPI.cpu()
                target_PPI = target_PPI.data[0].numpy().astype(np.float32)
                target_PPI = target_PPI * 255.
                target_PPI[target_PPI < 0] = 0
                target_PPI[target_PPI > 255.] = 255.

                im_gt_y = im_gt_y * 255.
                im_gt_y[im_gt_y < 0] = 0
                im_gt_y[im_gt_y > 255.] = 255.

                PPI_psnr_predicted = compute_PSNR(estimated_PPI[0,:,:],target_PPI[0,:,:])
                print("PSNR_PPI      =",PPI_psnr_predicted)
                psnr_predicted = compute_PSNR(im_gt_y.transpose(2, 1, 0),estimated_Demosaic.transpose(2, 1, 0))
                print("PSNR_Demosaic =", psnr_predicted)
                ssim_predicted_our = compute_ssim(im_gt_y.transpose(2, 1, 0), estimated_Demosaic.transpose(2, 1, 0))
                print("SSIM_Demosaic =", ssim_predicted_our)
                sam_predicted = compute_sam(im_gt_y.transpose(2, 1, 0), estimated_Demosaic.transpose(2, 1, 0))
                print("SAM_Demosaic  =", sam_predicted)

                kind = image_name[:-4]
                kind_dir = os.path.join('test_demosaic_result/' + kind + '/')
                os.makedirs(kind_dir, exist_ok=True)
                PPI_path = os.path.join(kind_dir + '/estimated_PPI.png')
                cv2.imwrite(PPI_path,estimated_PPI[0,:,:])
                PPI_real_path = os.path.join(kind_dir + '/real_PPI.png')
                cv2.imwrite(PPI_real_path, target_PPI[0,:,:])
                for channel in range(16):
                    demosaic_path = os.path.join(kind_dir + '/estimated_channel_'+str(channel)+'.png')
                    cv2.imwrite(demosaic_path, estimated_Demosaic[channel, :, :])

                    demosaic_real_path = os.path.join(kind_dir + '/real_channel_' + str(channel) + '.png')
                    cv2.imwrite(demosaic_real_path, im_gt_y[channel, :, :])

                demosaic_real_path = os.path.join(kind_dir + '/raw.png')
                cv2.imwrite(demosaic_real_path, raw[0,:,:])

                avg_PPI_psnr_predicted += PPI_psnr_predicted
                avg_psnr_predicted += psnr_predicted
                avg_sam_predicted += sam_predicted
                avg_ssim_predicted_our += ssim_predicted_our

                del estimated_PPI
                del raw
                del im_input

print("Dataset   :", opt.dataset)
print('sample_num:',sample_num)
print("PPI_PSNR_avg_predicted=", avg_PPI_psnr_predicted/sample_num)
print("PSNR    _avg_predicted=", avg_psnr_predicted / sample_num)
print("SSIM    _avg_predicted=", avg_ssim_predicted_our / sample_num)
print("SAM     _avg_predicted=", avg_sam_predicted / sample_num)
