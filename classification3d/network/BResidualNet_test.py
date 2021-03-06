import torch
from classification3d.network.BResidualNet import ClassificationNet


def test_DenseNet():
  batch_size, in_channel, dim_z, dim_y, dim_x = 2, 1, 32, 32, 32
  out_channel = 3

  input_tensor = torch.randn([batch_size, in_channel, dim_z, dim_y, dim_x])
  net = ClassificationNet(in_channel, out_channel, [dim_z, dim_y, dim_x])
  out = net.forward(input_tensor)
  assert out.shape[0] == batch_size and out.shape[1] == out_channel


if __name__ == "__main__":
  test_DenseNet()