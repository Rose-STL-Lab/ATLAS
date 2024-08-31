###########################################################################
#CGNet: A Light-weight Context Guided Network for Semantic Segmentation
#Paper-Link: https://arxiv.org/pdf/1811.08201.pdf
#This is taken from their implementation, we do not claim credit for this.
###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class GLConv2d(nn.Module):
    def __init__(self, equivariant, n_in, n_out, k_size, stride=1, bias=True, groups=1, dilation=1):
        super().__init__()

        if equivariant:
            self.weight = nn.Parameter(torch.empty(n_out, n_in // groups))
        else:
            self.weight = nn.Parameter(torch.empty(n_out, n_in // groups, *k_size))

        if bias:
            self.b = nn.Parameter(torch.empty(n_out))
        else:
            self.bias = None

        self.equivariant = equivariant
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.k_size = k_size

    def forward(self, x):
        if self.equivariant:
            weight = self.weight.unsqueeze(2).unsqueeze(3).expand(*self.weight.shape, *self.k_size)
        else:
            weight = self.weight

        return torch.nn.functional.conv2d(
            x, weight,
            bias=self.bias,
            stride=self.stride,
            groups=self.groups,
            dilation=self.dilation
        )

class Wrap(torch.nn.Module):

    def __init__(self, padding):
        super(Wrap, self).__init__()
        self.p = padding

    def forward(self, x):
        # creating the circular padding
        return F.pad(x, (self.p,)*4 , mode='circular')

class ConvBNPReLU(nn.Module):
    def __init__(self, equivariant, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.padding = Wrap(padding=padding)
        self.conv = GLConv2d(equivariant, nIn, nOut, (kSize, kSize), stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.padding(input)
        output = self.conv(output)
        output = self.bn(output)
        output = self.act(output)
        return output


class BNPReLU(nn.Module):
    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output

class ConvBN(nn.Module):
    def __init__(self, equivariant, nIn, nOut, kSize, stride=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.padding = Wrap(padding=padding)
        self.conv = GLConv2d(equivariant, nIn, nOut, (kSize, kSize), stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.padding(input)
        output = self.conv(output)
        output = self.bn(output)
        return output

class Conv(nn.Module):
    def __init__(self, equivariant, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.padding = Wrap(padding=padding)
        self.conv = GLConv2d(equivariant, nIn, nOut, (kSize, kSize), stride=stride, bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.padding(input)
        output = self.conv(output)
        return output

class ChannelWiseConv(nn.Module):
    def __init__(self, equivariant, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.padding = Wrap(padding=padding)
        self.conv = GLConv2d(equivariant, nIn, nOut, (kSize, kSize), stride=stride, groups=nIn, bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.padding(input)
        output = self.conv(output)
        return output
      
class DilatedConv(nn.Module):
    def __init__(self, equivariant, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.padding = Wrap(padding=padding)
        self.conv = GLConv2d(equivariant, nIn, nOut, (kSize, kSize), stride=stride, bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.padding(input)
        output = self.conv(output)
        return output

class ChannelWiseDilatedConv(nn.Module):
    def __init__(self, equivariant, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.padding = Wrap(padding=padding)
        self.conv = GLConv2d(equivariant, nIn, nOut, (kSize, kSize), stride=stride, groups=nIn, bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.padding(input)
        output = self.conv(output)
        return output

class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """
    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ContextGuidedBlock_Down(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """
    def __init__(self, equivariant, nIn, nOut, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        self.conv1x1 = ConvBNPReLU(equivariant, nIn, nOut, 3, 2)  #  size/2, channel: nIn--->nOut
        
        self.F_loc = ChannelWiseConv(equivariant, nOut, nOut, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(equivariant, nOut, nOut, 3, 1, dilation_rate)
        
        self.bn = nn.BatchNorm2d(2*nOut, eps=1e-3)
        self.act = nn.PReLU(2*nOut)
        self.reduce = Conv(equivariant, 2*nOut, nOut,1,1)  #reduce dimension: 2*nOut--->nOut
        
        self.F_glo = FGlo(nOut, reduction)    

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur],1)  #  the joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)     #channel= nOut
        
        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature

        return output


class ContextGuidedBlock(nn.Module):
    def __init__(self, equivariant, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, 
           add: if true, residual learning
        """
        super().__init__()
        n= int(nOut/2)
        self.conv1x1 = ConvBNPReLU(equivariant, nIn, n, 1, 1)  #1x1 Conv is employed to reduce the computation
        self.F_loc = ChannelWiseConv(equivariant, n, n, 3, 1) # local feature
        self.F_sur = ChannelWiseDilatedConv(equivariant, n, n, 3, 1, dilation_rate) # surrounding context
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo= FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        
        joi_feat = torch.cat([loc, sur], 1) 

        joi_feat = self.bn_prelu(joi_feat)

        output = self.F_glo(joi_feat)  #F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output  = input + output
        return output

class InputInjection(nn.Module):
    def __init__(self, downsamplingRatio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, downsamplingRatio):
            self.pool.append(Wrap(padding=1))
            self.pool.append(nn.AvgPool2d(3, stride=2))
    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input   
