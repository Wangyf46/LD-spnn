import os
import sys
sys.path.insert(0, '/home/wangyf/codes/LD-spnn')
import time
import cv2
import argparse
import ipdb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from lib.model import UNet_Upsampling
from lib.DIV2K import DIV2K
from lib.utils import *



os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def test_net(args, net):
    DATE = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    dir_LD = os.path.join('output', DATE, 'LD')
    dir_Iout = os.path.join('output', DATE, 'Iout')
    if not os.path.isdir(dir_LD):
        os.makedirs(dir_LD)
    if not os.path.isdir(dir_Iout):
        os.makedirs(dir_Iout)

    f = open(os.path.join('output', DATE, args.bl + '-BMA' + 'LDNNs-record.txt'), 'w')

    listDataset = DIV2K(args)

    test_loader = DataLoader(listDataset,
                              batch_size=args.bz,
                              pin_memory=True)

    net.eval()
    psnrs, ssims, cds, cr1s, cr2s = 0, 0, 0, 0, 0
    for i_batch, (Iin_transform, Icp_transform, BL_transform, name) in enumerate(test_loader):
        Iin_cpu = np.uint8(Iin_transform.squeeze(0).numpy().transpose((1, 2, 0)))
        Icp = Icp_transform.cuda()
        BL = BL_transform.unsqueeze(1).cuda()
        LD = net(BL)
        Iout = get_Iout(Icp, LD)

        LD_name = os.path.join(dir_LD, name[0])
        LD_cpu = np.uint8(LD.detach().squeeze(0).squeeze(0).cpu().numpy() * 255)
        cv2.imwrite(LD_name, LD_cpu)

        Iout_name = os.path.join(dir_Iout, name[0])
        Iout_cpu = np.uint8(Iout.detach().squeeze(0).cpu().numpy().transpose((1, 2, 0)) * 255)
        cv2.imwrite(Iout_name, Iout_cpu)

        psnr = get_PSNR(Iin_cpu, Iout_cpu)
        ssim = get_SSIM(Iin_cpu, Iout_cpu)  # 0.0-1.0
        cd = get_ColorDifference(Iin_cpu, Iout_cpu)  # 0.0
        cr1, cr2 = get_Contrast(Iin_cpu, Iout_cpu)

        psnrs += psnr
        ssims += ssim
        cds += cd
        cr1s += cr1
        cr2s += cr2

        print_str = 'Index: [{0}]'.format(name[0])
        print_str += 'PSNR: {0}'.format(psnr)
        print_str += 'SSIM: {0}'.format(ssim)
        print_str += 'CD: {0}'.format(cd)
        print_str += 'CR: [{0}/{1}]\t'.format(cr1, cr2)

        ## 打印到文件
        log_print(print_str, f, color="blue", attrs=["bold"])

    print(psnrs / i_batch, ssims / i_batch, cds / i_batch, cr1s / i_batch, cr2s / i_batch)
    print(psnrs / i_batch, ssims / i_batch, cds / i_batch, cr1s / i_batch, cr2s / i_batch, file=f)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="LD-spnn Test")
    parser.add_argument('--period', default='test')
    parser.add_argument('--path', default='/home/wangyf/datasets')
    parser.add_argument('--bz', default=1)
    parser.add_argument('--gpu', default=True)
    parser.add_argument('--pre',
                        default='/home/wangyf/codes/LD-spnn/checkpoints/2019-07-25-13-59/7_itr9600.pth',
                        help='load file model')
    parser.add_argument('--base_size', type=int, default=[1080, 1920])
    parser.add_argument('--block_size', type=int, default=[9, 16])
    parser.add_argument('--bl', type=str, default='LUT')
    parser.add_argument('--cp', type=str, default='LDNN')
    args = parser.parse_args()

    ## 固定随机种子
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    net = UNet_Upsampling(1, 1)

    if args.pre:
        net.load_state_dict(torch.load(args.pre))
        print('Model loaded from {}'.format(args.pre))

    if args.gpu:

        net.cuda()

    test_net(args, net)