import os
import sys
sys.path.insert(0, '/home/wangyf/codes/LD-spnn')
import time
import argparse
import ipdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from lib.model import UNet_Upsampling
from lib.DIV2K import DIV2K
from lib.utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def train_net(args, net):
    DATE = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    dir_log = os.path.join('log', DATE)
    dir_checkpoint = os.path.join('checkpoints/', DATE)
    if not os.path.isdir(dir_log):
        os.makedirs(dir_log)
    log_file = open(os.path.join(dir_log, 'record.txt'), 'w')
    tblogger = SummaryWriter(dir_log)

    listDataset = DIV2K(args)

    train_loader = DataLoader(listDataset,
                              batch_size=args.bz,
                              shuffle=True,
                              pin_memory=True)

    optimizer = optim.Adam(net.parameters(),
                           lr=args.lr,
                           betas=(0.9, 0.999),
                           eps=1e-08)

    criterion = nn.MSELoss(size_average=True)

    itr = 0
    max_itr = args.epochs * len(train_loader)
    print(itr, max_itr, len(train_loader))
    net.train()
    for epoch in range(args.epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))

        data_time = AverageMeter()
        batch_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()

        for i_batch, (Iin_transform, Icp_transform, BL_transform, name) in enumerate(train_loader):
            data_time.update(time.time() - end)                 # measure batch_size data loading time
            now_lr = adjust_lr(optimizer, epoch, args.lr)

            Iin = Iin_transform.cuda()                          # torch.float32, [0.0-255.0]
            Icp = Icp_transform.cuda()                          # torch.float32, [0.0-255.0]
            BL = BL_transform.unsqueeze(1).cuda()               # torch.float32-[0.0-1.0]

            LD = net(BL)                                        # torch.float32-[0.0-1.0]

            Iout = get_Iout(Icp, LD)                           # torch.float32-[0.0-1.0]

            loss = criterion(Iout, Iin / 255.0)
            losses.update(loss.item(), args.bz)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            print_str = 'Epoch: [{0}/{1}]\t'.format(epoch, args.epochs)
            print_str += 'Batch: [{0}]/{1}\t'.format(i_batch + 1, listDataset.__len__() // args.bz)
            print_str += 'LR: {0}\t'.format(now_lr)
            print_str += 'Data time {data_time.cur:.3f}({data_time.avg:.3f})\t'.format(data_time=data_time)
            print_str += 'Batch time {batch_time.cur:.3f}({batch_time.avg:.3f})\t'.format(batch_time=batch_time)
            print_str += 'Loss {loss.cur:.4f}({loss.avg:.4f})\t'.format(loss=losses)
            log_print(print_str, log_file, color="green", attrs=["bold"])

            tblogger.add_scalar('loss', losses.avg, itr)
            tblogger.add_scalar('lr', now_lr, itr)

            end = time.time()
            itr += 1
        if not os.path.isdir(dir_checkpoint):
            os.makedirs(dir_checkpoint)
        save_path = os.path.join(dir_checkpoint, '%s_itr%d.pth' % (epoch, itr))
        torch.save(net.state_dict(), save_path)
        print('%s has been saved' % save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LDNN Train")
    parser.add_argument('--period', default='train')
    parser.add_argument('--path', default='/home/wangyf/datasets')
    parser.add_argument('--epochs', default=8)                        ## 50
    parser.add_argument('--bz', default=2)
    parser.add_argument('--lr',  default=0.0001)
    parser.add_argument('--gpu', default=True)
    parser.add_argument('--pre',
                        default='/home/wangyf/codes/LD-spnn/checkpoints/2019-07-24-22-06/2019-07-24-22-06_42_itr51600.pth',
                        help='load file model')
    parser.add_argument('--base_size', type=int, default=[1080, 1920])
    parser.add_argument('--block_size', type=int, default=[9, 16])
    parser.add_argument('--bl', type=str, default='LUT')
    parser.add_argument('--cp', type=str, default='unlinear')
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
    try:
        train_net(args, net)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
