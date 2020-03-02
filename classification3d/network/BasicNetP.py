import torch.nn as nn
from collections import OrderedDict
from classification3d.network.module.weight_init import kaiming_weight_init


def parameters_init(net):
    net.apply(kaiming_weight_init)


class BasicBlock(nn.Module):
    """ downsample block of classification net """

    def __init__(self, in_channels):
        super(BasicBlock, self).__init__()
        out_channels = in_channels * 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.act(self.bn(self.conv(input)))
        return out


class InputBlock(nn.Module):
    """ input block of basic-net """

    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.act(self.bn(self.conv(input)))
        return out


class ClassificationNet(nn.Module):

    def __init__(self, in_channels, class_num, input_size):
        super(ClassificationNet, self).__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.pool_32 = nn.MaxPool3d(2, 2, 0)
        self.down_32 = BasicBlock(16)
        self.pool_64 = nn.MaxPool3d(2, 2, 0)
        self.down_64 = BasicBlock(32)
        self.pool_128 = nn.MaxPool3d(2, 2, 0)
        self.down_128 = BasicBlock(64)
        self.pool_256 = nn.MaxPool3d(2, 2, 0)
        self.down_256 = BasicBlock(128)

        reduce = OrderedDict()
        reduce['fc1'] = nn.Linear(256 * (input_size[0]//self.max_stride()) *
                                  (input_size[1]//self.max_stride()) *
                                  (input_size[2]//self.max_stride()), 256)
        reduce['relu'] = nn.ReLU(inplace=True)
        reduce['fc2'] = nn.Linear(256, class_num)
        reduce['softmax'] = nn.Softmax(dim=1)
        self.layer = nn.Sequential(reduce)

    def forward(self, input):
        out16 = self.in_block(input)
        out32 = self.down_32(self.pool_32(out16))
        out64 = self.down_64(self.pool_64(out32))
        out128 = self.down_128(self.pool_128(out64))
        out256 = self.down_256(self.pool_256(out128))
        out = out256.view(out256.size(0), -1)
        out = self.layer(out)
        return out

    @staticmethod
    def max_stride():
        return 16

