import torch.nn as nn
from classification3d.network.module.weight_init import kaiming_weight_init
from classification3d.network.module.vnet_inblock import InputBlock
from classification3d.network.module.vnet_downblock import DownBlock


def parameters_init(net):
    net.apply(kaiming_weight_init)


class ClassificationNet(nn.Module):

    def __init__(self, in_channels, class_num, input_size):
        super(ClassificationNet, self).__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1, compression=False)
        self.down_64 = DownBlock(32, 2, compression=True)
        self.down_128 = DownBlock(64, 3, compression=True)
        self.down_256 = DownBlock(128, 3, compression=True)
        self.down_512 = DownBlock(256, 3, compression=True)

        self.global_pooling = nn.AvgPool3d((input_size[0] // 32, input_size[1] // 32, input_size[2] // 32), 1, 0)
        self.fc = nn.Linear(512, class_num)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, input_tensor):
        out16 = self.in_block(input_tensor)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)
        out512 = self.down_512(out256)
        output = self.global_pooling(out512)
        output = output.view(output.size(0), -1)
        output = self.soft_max(self.fc(output))
        return output

    @staticmethod
    def max_stride():
        return 32

