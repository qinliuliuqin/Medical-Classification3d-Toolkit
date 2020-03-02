import torch.nn as nn
from collections import OrderedDict
from classification3d.network.module.weight_init import kaiming_weight_init
from classification3d.network.module.vnet_inblock import InputBlock
from classification3d.network.module.vnet_downblock import DownBlock


def parameters_init(net):
    net.apply(kaiming_weight_init)


class ClassificationNet(nn.Module):

    def __init__(self, in_channels, class_num, input_size):
        super(ClassificationNet, self).__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1)
        self.down_64 = DownBlock(32, 2)
        self.down_128 = DownBlock(64, 3)
        self.down_256 = DownBlock(128, 3)

        reduce = OrderedDict()
        reduce['fc1'] = nn.Linear(256 * (input_size[0] // self.max_stride()) *
                                  (input_size[1] // self.max_stride()) *
                                  (input_size[2] // self.max_stride()), 256)
        reduce['relu'] = nn.ReLU(inplace=True)
        reduce['fc2'] = nn.Linear(256, class_num)
        reduce['softmax'] = nn.Softmax(dim=1)
        self.layer = nn.Sequential(reduce)

    def forward(self, input):
        out16 = self.in_block(input)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)
        out = out256.view(out256.size(0), -1)
        out = self.layer(out)
        return out

    @staticmethod
    def max_stride():
        return 16

