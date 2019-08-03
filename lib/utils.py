import cv2
import ipdb
import numpy as np
from termcolor import cprint
from PIL import Image
from torchvision import transforms
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur, n=1):
        self.cur = cur
        self.sum += cur * n
        self.count += n
        self.avg = self.sum / self.count



def get_BL(block, mean='avg'):
    if mean == 'max':
        BL = np.max(block)
    elif mean == 'avg':
        BL = np.mean(block)
    elif mean == 'LUT':
        I_max = np.max(block)
        I_avg = np.mean(block)
        diff = I_max - I_avg
        BL = I_avg + 0.50 * (diff + diff ** 2 / 255)
    BL = np.where(BL < 0, 0, BL)
    BL = np.where(BL > 255, 255, BL)
    return BL


def LocalDimming(img, args):
    Iin = np.float32(img)                         # numpy-float32-[0.0, 255.0]
    h = args.base_size[0] / args.block_size[0]
    w = args.base_size[1] / args.block_size[1]
    BL = np.zeros(args.block_size, dtype='float32')
    B, G, R = cv2.split(Iin)
    R_G = np.where(R > G, R, G)
    gray = np.where(R_G > B, R_G, B)              # float32
    for i in range(args.block_size[0]):
        x1 = int(h * i)
        x2 = int(h * (i + 1))
        for j in range(args.block_size[1]):
            y1 = int(w * j)
            y2 = int(w * (j + 1))
            block = gray[x1:x2, y1:y2]
            BL[i][j] = get_BL(block, mean=args.bl)
    # BL= np.uint8(BL)                             # uint8
    return BL


def get_LD(BL, K):
    LD = smoothBL_BMA(BL, K).astype(np.float32)
    # cv2.namedWindow('BL', cv2.WINDOW_NORMAL)
    # cv2.imshow('BL', np.uint8(LD))
    # cv2.waitKey(0)
    return LD


def smoothBL_BMA(BL_init, K):
    ## TODO
    a_R, b_R, c_R, d_R = 0.4, 0.2, 0.2, 0.2
    a_G, b_G, c_G, d_G = 0.38, 0.15, 0.12, 0.1
    a_B, b_B, c_B, d_B = 0.38, 0.11, 0.08, 0.06

    for t in range(K):
        Height, Width = BL_init.shape
        BL_mirr = np.zeros((Height+2, Width+2))

        # mid
        for i in range(Height):
            for j in range(Width):
                BL_mirr[i+1][j+1] = BL_init[i][j]

        # Up and Down
        for j in range(Width):
            BL_mirr[0][j+1] = BL_init[0][j]
            BL_mirr[Height+1][j+1] = BL_init[Height-1][j]

        # Left and Right
        for i in range(Height):
            BL_mirr[i+1][0] = BL_init[i][0]
            BL_mirr[i+1][Width+1] = BL_init[i][Width-1]

        # four
        BL_mirr[0][0] = BL_init[0][0]
        BL_mirr[0][Width+1] = BL_init[0][Width-1]
        BL_mirr[Height+1][0] = BL_init[Height-1][0]
        BL_mirr[Height+1][Width+1] = BL_init[Height-1][Width-1]


        # RED Local
        BL_blur = BL_mirr
        BL_blur[1][1] = a_R * BL_mirr[1][1] + b_R * BL_mirr[1][2] + \
                        c_R * BL_mirr[2][1] + d_R * BL_mirr[2][2]
        BL_blur[1][Width] = a_R * BL_mirr[1][Width] + b_R * BL_mirr[1][Width-1] + \
                            c_R * BL_mirr[2][Width] + d_R * BL_mirr[2][Width-1]
        BL_blur[Height][1] = a_R * BL_mirr[Height][1] + b_R * BL_mirr[Height][2] + \
                             c_R * BL_mirr[Height-1][1] + d_R * BL_mirr[Height-1][2]
        BL_blur[Height][Width] = a_R * BL_mirr[Height][Width] + b_R * BL_mirr[Height][Width-1] + \
                                 c_R * BL_mirr[Height-1][Width] + d_R * BL_mirr[Height-1][Width-1]

        # Green Local(left-right)
        for i in range(2, Height):
            BL_blur[i][1] = a_G * BL_mirr[i][1] + b_G * (BL_mirr[i-1][1] + BL_mirr[i+1][1]) + \
                            c_G * BL_mirr[i][2] + d_G * (BL_mirr[i-1][2] + BL_mirr[i+1][2])

            BL_blur[i][Width] = a_G * BL_mirr[i][Width] + b_G * (BL_mirr[i-1][Width] + BL_mirr[i+1][Width]) + \
                            c_G * BL_mirr[i][Width-1] + d_G * (BL_mirr[i-1][Width-1] + BL_mirr[i+1][Width-1])

        # Green Local(up-down)
        for j in range(2, Width):
            BL_blur[1][j] = a_G * BL_mirr[1][j] + b_G * (BL_mirr[1][j-1] + BL_mirr[1][j+1])+ \
                        c_G * BL_mirr[2][j] + d_G * (BL_mirr[2][j-1] + BL_mirr[2][j+1])
            BL_blur[Height][j] =  a_G * BL_mirr[Height][j] + b_G * (BL_mirr[Height][j-1] + BL_mirr[Height][j+1]) + \
                              c_G * BL_mirr[Height-1][j] + d_G * (BL_mirr[Height-1][j-1] + BL_mirr[Height-1][j+1])

        # BLUE block
        for i in range(2, Height):
            for j in range(2, Width):
                BL_blur[i][j] = a_B * BL_mirr[i][j] + b_B * (BL_mirr[i][j-1] + BL_mirr[i][j+1]) + \
                            c_B * (BL_mirr[i-1][j] + BL_mirr[i+1][j]) + \
                            d_B * (BL_mirr[i-1][j-1] + BL_mirr[i-1][j+1]+ BL_mirr[i+1][j-1]  + BL_mirr[i+1][j+1])
        h, w =  BL_blur.shape
        BL_blur_2x = cv2.resize(BL_blur, (2*w, 2*h))
        BL_init = BL_blur_2x

    LD = cv2.resize(BL_blur_2x, (1920, 1080))
    LD = np.where(LD<0, 0, LD)
    LD = np.where(LD>255, 255, LD)
    return LD


def get_Icp(Iin_transform, LD_transform, means='Linear'):
    Iins = Iin_transform.numpy().transpose((0, 2, 3, 1))
    LDs = LD_transform.squeeze(1).detach().cpu().numpy() * 255.0
    Icps = []
    for index in range(Iins.shape[0]):
        Iin = Iins[index]
        LD = LDs[index]
        if means == 'linear':
            Icp = getCP_linear(Iin, LD)
        elif means == 'unlinear':
            Icp = getCP_unlinear(Iin, LD)    ## block effect
        elif means == '2steps':
            Icp = getCP_2steps(Iin, LD)
        elif means == 'log':
            Icp = getCP_log(Iin, LD)
        # elif means == 'LDNN':
        Icps.append(Icp)
    Icps_transform = torch.from_numpy(np.array(Icps).transpose((0, 3, 1, 2)))
    return Icps_transform



## TODO, only YUV, RGB or Iin is not
def getCP_linear(Iin, LD):
    B, G, R = cv2.split(Iin)
    Y, U, V = rgbToyuv(R, G, B)
    # LD_max = np.max(LD)
    LD_max = 255.0
    if LD.any() != 0:
        K = LD_max / LD
    else:
        K = 0
    Y1 = K * Y
    U1 = K * (U - 128) + 128
    V1 = K * (V - 128) + 128
    R1, G1, B1 = yuvTorgb(Y1, U1, V1)
    Icp = cv2.merge([B1, G1, R1])
    return Icp


def getCP_unlinear(Iin, LD):
    B, G, R = cv2.split(Iin)
    Y, U, V = rgbToyuv(R, G, B)
    # LD_max = np.max(LD)
    LD_max = 255.0
    r = 2.2
    # for i in range(M):
    #     for j in range(N):
    #         if (LD[i][j] >= 0) and (LD[i][j] < 8):
    #             LD[i][j] = 0.4625 * LD[i][j] + 0.3
    #         else:
    #             LD[i][j] = 1.1984 * LD[i][j] - 5.592
    if LD.any() != 0:
        K = (LD_max / LD) ** (1 / r)
    else:
        K = 0

    Y1 = K * Y
    U1 = K * (U - 128) + 128
    V1 = K * (V - 128) + 128
    R1, G1, B1 = yuvTorgb(Y1, U1, V1)
    Icp = cv2.merge([B1, G1, R1])
    return Icp



def getCP_2steps(Iin, LD):
    Iin_max = np.max(Iin)
    B, G, R = cv2.split(Iin)
    Y, U, V = rgbToyuv(R, G, B)
    a = 0.005
    LD_max = 255.0
    gamma = 2.2

    ## improve cr
    Y1 = Iin_max / (1 + np.exp(a * (LD - Y)))
    Y1 = np.where(Y1 > 255, 255, Y1)

    ## enhance diaplay quality
    K2 = (LD / LD_max) ** (1.0 / gamma)
    Y2 = Y1 * np.log10(1 + Y * K2)

    ## ratio
    if Y1.any() != 0:
        K = Y2 / Y1
    else:
        K = Y2 / (Y1 + 0.0001)

    U2 = K * (U - 128) + 128
    V2 = K * (V - 128) + 128
    R1, G1, B1 = yuvTorgb(Y2, U2, V2)
    Icp = cv2.merge([B1, G1, R1])

    return Icp



def getCP_log(Iin, LD):
    B, G, R = cv2.split(Iin)
    Y, U, V = rgbToyuv(R, G, B)
    M, N = LD.shape
    # LD_max = np.max(LD)
    LD_max = 255.0
    gamma = 2.2

    # for i in range(M):
    #     for j in range(N):
    #         if (LD[i][j] >= 0) and (LD[i][j] < 8):
    #             LD[i][j] = 0.4625 * LD[i][j] + 0.3
    #         else:
    #             LD[i][j] = 1.1984 * LD[i][j] - 5.592

    if LD.any() != 0:
        K = (LD_max / LD) ** (1 / gamma)
    else:
        K = 0
    K1 = np.log2(1 + K)

    Y1 = K1 * Y
    U1 = K1 * (U - 128) + 128
    V1 = K1 * (V - 128) + 128
    R1, G1, B1 = yuvTorgb(Y1, U1, V1)
    Icp = cv2.merge([B1, G1, R1])
    return Icp


def get_Iout(Icp, LD):
    Iout = Icp * LD / 255.0
    return Iout


def adjust_lr(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 20 every 10 epochs"""
    lr = lr * (0.2 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def log_print(text, log_file, color = None, on_color = None, attrs = None):
    print(text, file=log_file)
    if cprint is not None:
        cprint(text, color = color, on_color = on_color, attrs = attrs)
    else:
        print(text)


def get_PSNR(target, ref):
    MSE = np.mean((target - ref) ** 2)
    PSNR = 20 * np.log10(255 / np.sqrt(MSE))
    return PSNR



def get_SSIM(target, ref):
    target = np.float32(target)
    ref = np.float32(ref)

    k1, k2 = 0.01, 0.03
    L = 255
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    c3 = c2 / 2
    alpha, beta, gamma = 1, 1, 1

    ux, uy = np.mean(target), np.mean(ref)
    vx, vy = np.var(target), np.var(ref)  # variance
    sdx, sdy = np.std(target), np.std(ref)  # sd
    Covariance = np.sum(target * ref) / (target.shape[0] * target.shape[1] * 3) - ux * uy

    Luminance = (2 * ux * uy + c1) / (ux ** 2 + uy ** 2 + c1)
    Contrast = (2 * sdx * sdy + c2) / (vx + vy + c2)
    Structure = (Covariance + c3) / (sdx * sdy + c3)

    SSIM = (Luminance ** alpha) * (Contrast ** beta) * (Structure ** gamma)
    return SSIM


## RGB-->XYX--->LAB
def get_ColorDifference(target, ref):
    B1, G1, R1 = cv2.split(target)
    R1_gamma, G1_gamma, B1_gamma = gamma(R1/255.0, G1/255.0, B1/255.0)
    X1, Y1, Z1 = rgbToxyz(R1_gamma, G1_gamma, B1_gamma)
    L1, a1, b1 = xyzTolab(X1, Y1, Z1)

    B2, G2, R2 = cv2.split(ref)
    R2_gamma, G2_gamma, B2_gamma = gamma(R2 / 255.0, G2 / 255.0, B2 / 255.0)
    X2, Y2, Z2 = rgbToxyz(R2_gamma, G2_gamma, B2_gamma)
    L2, a2, b2 = xyzTolab(X2, Y2, Z2)

    CD = np.mean(np.sqrt((L2-L1)**2 + (a2-a1)**2 + (b1-b2)**2))
    return CD



def get_H(Y):
    Height, Width = Y.shape
    p = 0.1 * Height * Width
    q = 0.9 * Height * Width
    Y_max = np.max(Y)
    Y_min = np.min(Y)
    total = 0
    for H_10 in range(Y_min, Y_max+1):
        num = np.sum(Y == H_10)
        total += num
        if total > p:
            break
    for H_90 in range(Y_min, Y_max + 1):
        num = np.sum(Y == H_90)
        total += num
        if total > q:
            break

    return H_10, H_90



## CR
def get_Contrast(Iin, Iout):
    Bin, Gin, Rin = cv2.split(Iin)
    Yin, _, _ = rgbToyuv(Rin, Gin, Bin)
    Hin_10, Hin_90 = get_H(np.uint8(Yin))

    Bout, Gout, Rout = cv2.split(Iout)
    Yout, _, _ = rgbToyuv(Rout, Gout, Bout)
    Hout_10, Hout_90 = get_H(np.uint8(Yout))

    if Hin_10 != 0:
        cr1 = Hin_90 / Hin_10
    else:
        cr1 = 1000

    if Hout_10 != 0:
        cr2 = Hout_90 / Hout_10
    else:
        cr2 = 1000

    return cr1, cr2


def rgbToyuv(R, G, B):
    R = np.where(R>255, 255, R)
    R = np.where(R<0, 0, R)
    G = np.where(G>255, 255, G)
    G = np.where(G<0, 0, G)
    B = np.where(B>255, 255, B)
    B = np.where(B<0, 0, B)

    Y = 0.2989 * R + 0.5866 * G + 0.1145 * B
    U = -0.1688 * R - 0.3312 * G + 0.5 * B + 128
    V = 0.5 * R - 0.4184 * G - 0.0816 * B + 128

    Y = np.where(Y > 255, 255, Y)
    Y = np.where(Y < 0, 0, Y)
    U = np.where(U>255, 255, U)
    U = np.where(U<0, 0, U)
    V = np.where(V>255, 255, V)
    V = np.where(V<0, 0, V)

    return Y, U, V



def yuvTorgb(Y, U, V):
    Y = np.where(Y > 255, 255, Y)
    Y = np.where(Y < 0, 0, Y)
    U = np.where(U>255, 255, U)
    U = np.where(U<0, 0, U)
    V = np.where(V>255, 255, V)
    V = np.where(V<0, 0, V)

    R = Y + 1.4021 * (V - 128)
    G = Y - 0.3456 * (U - 128)- 0.7145 * (V - 128)
    B = Y + 1.771 * (U - 128)

    R = np.where(R>255, 255, R)
    R = np.where(R<0, 0, R)
    G = np.where(G>255, 255, G)
    G = np.where(G<0, 0, G)
    B = np.where(B>255, 255, B)
    B = np.where(B<0, 0, B)

    return R, G, B



def rgbToxyz(R, G, B):
    X = 0.4124 * R + 0.3576 * G + 0.1805 * B
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    Z = 0.0193 * R + 0.1192 * G + 0.9505 * B
    return X, Y, Z



def xyzTolab(X, Y, Z):
    v1 = 1.0/3
    v2 = (6.0/29) ** 3
    v3 = (29.0/6) ** 2
    v4 = 4.0/29
    Xn = 95.047
    Yn = 100.0
    Zn = 108.883

    X1 = Y / Yn
    Y1 = X / Xn
    Z1 = Z / Zn

    Xf = np.where(X1 > v2, X1 ** v1, v1 * v3 * X1 + v4)
    Yf = np.where(Y1 > v2, Y1 ** v1, v1 * v3 * Y1 + v4)
    Zf = np.where(Z1 > v2, Z1 ** v1, v1 * v3 * Z1 + v4)

    L = 116 * Yf - 16
    a = 500 * (Xf - Yf)
    b = 200 * (Yf - Zf)

    return L, a, b



def gamma(R, G, B):
    R1 = np.where(R > 0.04045, ((R + 0.055) / 1.055) ** 2.4, R / 12.92)
    G1 = np.where(G > 0.04045, ((G + 0.055) / 1.055) ** 2.4, G / 12.92)
    B1 = np.where(B > 0.04045, ((B + 0.055) / 1.055) ** 2.4, B / 12.92)
    return R1, G1, B1