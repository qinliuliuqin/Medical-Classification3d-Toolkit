import torch
import torch.nn as nn


def parameters_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


class InputBlock(nn.Module):
    """ input block of Dense net """

    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()

        self.convolution = nn.Conv3d(in_channels, out_channels, kernel_size=7, padding=3, bias=False)
        self.batch_normal = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.activation(self.batch_normal(self.convolution(input_tensor)))
        return out


class DenseBottlePart(nn.Module):
    """Basic part for Dense block."""

    def __init__(self, input_channel, output_channel):
        super(DenseBottlePart, self).__init__()

        self.batch_norm1 = nn.BatchNorm3d(input_channel)
        self.activation1 = nn.ReLU(inplace=True)
        self.convolution1 = nn.Conv3d(input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.batch_norm2 = nn.BatchNorm3d(output_channel)
        self.activation2 = nn.ReLU(inplace=True)
        self.convolution2 = nn.Conv3d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input_tensor):
        output = self.convolution1(self.activation1(self.batch_norm1(input_tensor)))
        output = self.convolution2(self.activation2(self.batch_norm2(output)))
        return output


class DenseBlock(nn.Module):
    """Dense block"""

    def __init__(self, input_channel, growth_rate, dense_part_number):
        super(DenseBlock, self).__init__()

        self.dense_base_blocks = torch.nn.ModuleList()
        for i in range(dense_part_number):
            self.dense_base_blocks.append(DenseBottlePart(input_channel + growth_rate * i, growth_rate))

    def forward(self, input_tensor):
        mid = input_tensor
        for i in range(len(self.dense_base_blocks)):
            output_tensor = self.dense_base_blocks[i](mid)
            mid = torch.cat((mid, output_tensor), 1)
        return mid


class TransitionLayer(nn.Module):
    """Transition layer using after dense block"""

    def __init__(self, input_channels, output_channels):
        super(TransitionLayer, self).__init__()

        self.batch_norm = nn.BatchNorm3d(input_channels)
        self.activation = nn.ReLU(inplace=True)
        self.convolution = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pooling = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, input_tensor):
        output = self.convolution(self.activation(self.batch_norm(input_tensor)))
        output = self.pooling(output)
        return output


class ClassificationNet(nn.Module):
    """Dense net"""

    def __init__(self, in_channels, class_num, input_size, growth_rate=32, dense_convolution_num_list=[4, 4, 4, 4]):
        """
        :param in_channels: input channels
        :param class_num: class number
        :param input_size: input size
        :param growth_rate: growth rate for dense net
        :param dense_convolution_num_list: the number of convolution layer in each dense block
        """
        super(ClassificationNet, self).__init__()

        self.input_block = InputBlock(in_channels, 16)
        self.input_pooling = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # generate dense block
        layers = list()
        dense_block_number = len(dense_convolution_num_list)
        feature_num_output = 16

        for i in range(dense_block_number):
            dense_convolution_num = dense_convolution_num_list[i]
            layers.append(DenseBlock(feature_num_output, growth_rate, dense_convolution_num))
            feature_num_output = feature_num_output + growth_rate * dense_convolution_num
            if i != dense_block_number - 1:
                layers.append(TransitionLayer(feature_num_output, feature_num_output//2))
                feature_num_output = feature_num_output // 2
        self.ops = nn.Sequential(*layers)

        self.final_bn = nn.BatchNorm3d(feature_num_output)
        self.final_act = nn.ReLU(inplace=True)

        self.global_pooling = nn.AvgPool3d((input_size[0]//16, input_size[1]//16, input_size[2]//16), 1, 0)
        self.fc = nn.Linear(feature_num_output, class_num)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, input_tensor):
        output = self.input_pooling(self.input_block(input_tensor))
        output = self.ops(output)
        output = self.final_act(self.final_bn(output))
        output = self.global_pooling(output)
        output = output.view(output.size(0), -1)
        output = self.soft_max(self.fc(output))
        return output

    @staticmethod
    def max_stride():
        return 16