import numpy as np
from mindspore import nn
from mindspore import Parameter, Tensor
import mindspore.ops as ops
import mindspore.ops.operations as P
from utils import default_recurisive_init, KaimingNormal
import math
from mindspore.common import initializer as init
import mindspore as ms
class Discriminator(nn.Cell):
    def __init__(self, conv_dim=32, out_class=1):
        super(Discriminator, self).__init__()

        self.conv_dim =conv_dim
        n_class = out_class
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv3d(1, conv_dim, kernel_size=3, stride=2, padding=1, pad_mode="pad", has_bias=True)
        self.bn1 =nn.BatchNorm3d(conv_dim)

        self.conv2 = nn.Conv3d(conv_dim, conv_dim*2, kernel_size=3, stride=2, padding=1, pad_mode="pad", has_bias=True)
        self.bn2 = nn.BatchNorm3d(conv_dim*2)

        self.conv3 = nn.Conv3d(conv_dim*2, conv_dim*4, kernel_size=3, stride=2, padding=1, pad_mode="pad", has_bias=True)
        self.bn3 = nn.BatchNorm3d(conv_dim*4)

        self.conv4 = nn.Conv3d(conv_dim*4, conv_dim*8, kernel_size=3, stride=2, padding=1, pad_mode="pad", has_bias=True)
        self.bn4 = nn.BatchNorm3d(conv_dim*8) 

        self.conv5 = nn.Conv3d(conv_dim*8, conv_dim*16, kernel_size=3, stride=2, padding=1, pad_mode="pad", has_bias=True)
        self.bn5 = nn.BatchNorm3d(conv_dim*16) 

        self.conv6 = nn.Conv3d(conv_dim*16, n_class, kernel_size=3, stride=1, padding=0, pad_mode="pad", has_bias=True)

        # default_recurisive_init(self)
        # for _, cell in self.cells_and_names():
        #     if isinstance(cell, nn.Conv3d):
        #         cell.weight.set_data(init.initializer(KaimingNormal(a=math.sqrt(5), mode='fan_out',
        #                                                             nonlinearity='leaky_relu'),
        #                                               cell.weight.shape,
        #                                               cell.weight.dtype))

    def contruct(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        output = self.sigmoid(x.view(x.shape[0], -1))

        return output


class PAM(nn.Cell):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(np.zeros(1))
        self.batmatmul = ops.BatchMatMul()
        self.softmax = nn.Softmax()

    def construct(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W X Z)
            returns :
                out : attention value + input feature
                attention: B X (HxWxZ) X (HxWxZ)
        """
        m_batchsize, C, height, width, deep = x.shape
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*deep)
        proj_query = ops.permute(proj_query, (0, 2, 1))
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*deep)
        energy = self.batmatmul(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*deep)

        attention = ops.permute(attention, (0, 2, 1))
        out = self.batmatmul(proj_value, attention)

        out = out.view(m_batchsize, C, height, width, deep)
        out = self.gamma*out + x
        return out

class CAM(nn.Cell):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.chanel_in = in_dim
        self.batmatmul = ops.BatchMatMul()
        self.gamma = Parameter(np.zeros(1))
        self.softmax  = nn.Softmax()
        self.argmax = ops.ArgMaxWithValue()

    def construct(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W X Z)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, deep = x.shape
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1)
        proj_key = ops.permute(proj_key, (0, 2, 1))
        energy = self.batmatmul(proj_query, proj_key)
        energy_new = self.argmax()(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = self.batmatmul(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, deep)
        out = self.gamma*out + x
        return out

class Generator(nn.Cell):
    def __init__(self, conv_dim=8):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.relu = nn.ReLU()
        self.down_sampling = P.MaxPool3D(kernel_size=3, strides=2)
        self.pa = PAM(64)
        self.concat_op = ops.Concat(1)

        #Encoder
        self.tp_conv1 = nn.Conv3d(1, conv_dim, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)
        self.bn1 = nn.BatchNorm3d(conv_dim)
        self.tp_conv2 = nn.Conv3d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)
        self.bn2 = nn.BatchNorm3d(conv_dim)

        self.tp_conv3 = nn.Conv3d(conv_dim, conv_dim*2, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)
        self.bn3 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv4 = nn.Conv3d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)
        self.bn4 = nn.BatchNorm3d(conv_dim*2)  

        self.tp_conv5 = nn.Conv3d(conv_dim*2, conv_dim*4, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)
        self.bn5 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv6 = nn.Conv3d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)
        self.bn6 = nn.BatchNorm3d(conv_dim*4)  

        self.tp_conv7 = nn.Conv3d(conv_dim*4, conv_dim*8, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)
        self.bn7 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv8 = nn.Conv3d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)
        self.bn8 = nn.BatchNorm3d(conv_dim*8)  

        self.rbn = nn.Conv3d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)

        #Decoder
        self.tp_conv9 = nn.Conv3dTranspose(conv_dim*8, conv_dim*4, kernel_size=3, stride=2, padding=0, output_padding=1, pad_mode="pad", has_bias=True)
        self.bn9 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv10 = nn.Conv3d(conv_dim*8, conv_dim*4, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)
        self.bn10 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv11 = nn.Conv3d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)
        self.bn11 = nn.BatchNorm3d(conv_dim*4)

        self.tp_conv12 = nn.Conv3dTranspose(conv_dim*4, conv_dim*2, kernel_size=3, stride=2, padding=0, output_padding=(0, 1, 0), pad_mode="pad", has_bias=True)
        self.bn12 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv13 = nn.Conv3d(conv_dim*4, conv_dim*2, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)
        self.bn13 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv14 = nn.Conv3d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)
        self.bn14 = nn.BatchNorm3d(conv_dim*2)

        self.tp_conv15 = nn.Conv3dTranspose(conv_dim*2, conv_dim*1, kernel_size=3, stride=2, padding=0, output_padding=(1, 1, 1), pad_mode="pad", has_bias=True)
        self.bn15 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv16 = nn.Conv3d(conv_dim*2, conv_dim*1, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)
        self.bn16 = nn.BatchNorm3d(conv_dim*1)
        self.tp_conv17 = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)
        self.bn17 = nn.BatchNorm3d(conv_dim*1)


        self.tp_conv18 = nn.Conv3d(conv_dim*1, 1, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)

        self.convpa = nn.SequentialCell(nn.Conv3d(conv_dim*8, conv_dim*8, 3, padding=1, pad_mode="pad", has_bias=True),
                                   nn.BatchNorm3d(conv_dim*8),
                                   nn.ReLU())

    def construct(self, z):
        z = Tensor(z, ms.float32)
        h = self.tp_conv1(z)
        h = self.tp_conv2(self.relu(h))
        skip3 = h
        h = self.down_sampling(self.relu(skip3))


        h = self.tp_conv3(h)
        h = self.tp_conv4(self.relu(h))
        skip2 = h
        h = self.down_sampling(self.relu(skip2))

        h = self.tp_conv5(h)
        h = self.tp_conv6(self.relu(h))
        skip1 = h
        h = self.down_sampling(self.relu(skip1))        

        h = self.tp_conv7(h)
        h = self.tp_conv8(self.relu(h))
    
        # pa_feat = self.convpa(h)
        # pa_feat = self.pa(pa_feat)



        h = self.tp_conv9((self.relu(h)))
        h = self.concat_op((h, skip1))
        h = self.relu(h)
        h = self.relu(self.tp_conv10(h))
        h = self.relu(self.tp_conv11(h))

        h = self.tp_conv12(h)
        h = self.concat_op((h, skip2))
        h = self.relu(h)
        h = self.relu(self.tp_conv13(h))
        h = self.relu(self.tp_conv14(h))

        h = self.tp_conv15(h)
        h = self.concat_op((h, skip3))
        h = self.relu(h)
        h = self.relu(self.tp_conv16(h))
        h = self.relu(self.tp_conv17(h))

        h = self.tp_conv18(h)

        return h