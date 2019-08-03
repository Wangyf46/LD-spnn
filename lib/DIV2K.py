import os
import cv2
import ipdb
import torch
import numpy as np
from torch.utils.data import Dataset

from lib.utils  import *

class DIV2K(Dataset):
    def __init__(self, args):
        self.args = args
        if args.period == 'train':
            self.Iin_dir = os.path.join(args.path, 'DIV2K_train_HR_aug')
            self.Icp_dir = os.path.join(args.path, 'DIV2K_train_HR_aug_Icp')               ##TODO
            self.name_list = os.listdir(self.Iin_dir)

        else:
            self.Iin_dir = os.path.join(args.path, 'DIV2K_valid_HR_aug')
            self.Icp_dir = os.path.join(args.path, 'DIV2K_valid_HR_aug_Icp')               ##TODO
            self.name_list = os.listdir(self.Iin_dir)


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        Iin_file =  os.path.join(self.Iin_dir, name)
        Icp_file = os.path.join(self.Icp_dir, name)

        Iin = cv2.imread(Iin_file)                               # numpy-RGB-uint8-[0-255]
        Icp = cv2.imread(Icp_file)                               # numpy-RGB-uint8-[0-255]

        BL = LocalDimming(Iin, self.args)                        # numpy-float32-[0.0-255.0]

        Iin_transform = torch.from_numpy(Iin.transpose((2, 0, 1)).astype(np.float32))    # torch.float32-[0.0-255.0]
        Icp_transform = torch.from_numpy(Icp.transpose((2, 0, 1)).astype(np.float32))  # torch.float32-[0.0-255.0]
        BL_transform = torch.from_numpy(BL / 255.0)                                    # torch.float32-[0.0-1.0]

        return  Iin_transform, Icp_transform, BL_transform, name


