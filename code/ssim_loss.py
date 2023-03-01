from math import exp
import numpy as np
from mindspore import nn
from mindspore import Parameter, Tensor
import mindspore.ops as ops


class SSIM(nn.Cell):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()


    def construct(self, img1, img2):
        max_val = ops.ArgMaxWithValue(img1)
 
        min_val = ops.ArgMinWithValue(img1)

        L = max_val - min_val

        net = nn.SSIM(max_val=L, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

        (_, channel, _, _, slices) = img1.size()
        ssim_v = 0
        for s in range(slices):
            ssim_v += 1-net(img1[:, :, :, :, s], img2[:, :, :, :, s])

        return ssim_v/slices
