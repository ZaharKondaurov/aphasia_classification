import torch
from torch import nn
import math


class MaskedConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, mask=None, input_size=42):
        super(MaskedConv1d, self).__init__()

        # padding = max(0, (math.ceil((stride - 1) * in_channels + (kernel_size - 1) * dilation + 1 - stride) / stride - input_size))
        # padding = (stride * (input_size - 1) + kernel_size - input_size) // 2
        # padding = stride * (kernel_size - 1) * dilation // 2
        # if kernel_size % 2 == 0:
        #     padding -= 1
        # padding = ((kernel_size - 1) * dilation - stride + 1)// 2
        padding = math.ceil(((input_size - 1) * stride + 1 + dilation * (kernel_size - 1) - input_size) / 2)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding)

        # if mask is None:
        #     self.mask = torch.ones(kernel_size)
        # else:
        #     self.mask = mask.detach().clone()
        #
        # self.mask = nn.Parameter(self.mask, requires_grad=False)
        #
        # with torch.no_grad():
        #     self.conv.weight = torch.nn.Parameter(self.mask * self.conv.weight)
        #
        # self.conv.weight.register_hook(lambda x: x * self.mask)

    def forward(self, x):
        return self.conv(x)


class GatedActivation(nn.Module):

    def __init__(self):
        super(GatedActivation, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_sigmoid = self.sigmoid(x)
        x_tanh = self.tanh(x)
        return x_sigmoid * x_tanh * x  # x * ???


class GatedConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, mask=None, input_size=42):
        super(GatedConv1d, self).__init__()

        self.conv = MaskedConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, mask=mask,
                                 stride=stride, dilation=dilation, input_size=input_size)

        self.gate = GatedActivation()

    def forward(self, x):
        x = self.conv(x)
        x = self.gate(x)
        return x


class SwishNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, input_size=42, dropout_rate=0.1):
        super(SwishNet, self).__init__()

        kernel_size_3 = 3
        mask_3 = torch.ones(kernel_size_3)
        mask_3[kernel_size_3 // 2 + 1:] -= 1

        kernel_size_6 = 5
        mask_6 = torch.ones(kernel_size_6)
        mask_6[kernel_size_6 // 2:] -= 1

        self.gatedconv1_1 = GatedConv1d(in_channels=in_channels, out_channels=16, kernel_size=kernel_size_3,
                                        mask=mask_3, input_size=input_size)
        self.gatedconv1_2 = GatedConv1d(in_channels=in_channels, out_channels=16, kernel_size=kernel_size_6,
                                        mask=mask_6, input_size=input_size)

        self.dropout1 = nn.Dropout(dropout_rate)

        self.batchnorm1 = nn.BatchNorm1d(32)
        self.gatedconv2_1 = GatedConv1d(in_channels=32, out_channels=8, kernel_size=kernel_size_3, mask=mask_3,
                                        input_size=input_size)
        self.gatedconv2_2 = GatedConv1d(in_channels=32, out_channels=8, kernel_size=kernel_size_6, mask=mask_6,
                                        input_size=input_size)

        self.dropout2 = nn.Dropout(dropout_rate)

        self.batchnorm2 = nn.BatchNorm1d(16)
        self.gatedconv3_1 = GatedConv1d(in_channels=16, out_channels=8, kernel_size=kernel_size_3, mask=mask_3,
                                        input_size=input_size)
        self.gatedconv3_2 = GatedConv1d(in_channels=16, out_channels=8, kernel_size=kernel_size_6, mask=mask_6,
                                        input_size=input_size)

        self.dropout3 = nn.Dropout(dropout_rate)

        self.batchnorm3 = nn.BatchNorm1d(16)
        self.gatedconv4 = GatedConv1d(in_channels=16, out_channels=16, kernel_size=kernel_size_3, mask=mask_3, stride=3,
                                      input_size=input_size)

        self.dropout4 = nn.Dropout(dropout_rate)

        self.batchnorm4 = nn.BatchNorm1d(16)
        self.gatedconv5 = GatedConv1d(in_channels=16, out_channels=16, kernel_size=kernel_size_3, mask=mask_3, stride=2,
                                      input_size=input_size)

        self.batchnorm5 = nn.BatchNorm1d(16)
        self.dropout5 = nn.Dropout(dropout_rate)
        self.gatedconv6 = GatedConv1d(in_channels=16, out_channels=16, kernel_size=kernel_size_3, mask=mask_3, stride=2,
                                      input_size=input_size)

        self.dropout6 = nn.Dropout(dropout_rate)

        self.batchnorm6 = nn.BatchNorm1d(16)
        self.gatedconv7 = GatedConv1d(in_channels=16, out_channels=16, kernel_size=kernel_size_3, mask=mask_3, stride=2,
                                      input_size=input_size)  # 2

        self.batchnorm7 = nn.BatchNorm1d(16)
        self.dropout7 = nn.Dropout(dropout_rate)
        self.gateconv8 = GatedConv1d(in_channels=16, out_channels=32, kernel_size=kernel_size_3, mask=mask_3, stride=1,
                                     input_size=input_size)  # 2

        # self.gateconv9 = GatedConv1d(in_channels=80, out_channels=out_channels, kernel_size=1, input_size=input_size)

        self.batchnorm8 = nn.BatchNorm1d(32)
        self.conv9 = nn.Conv1d(in_channels=80, out_channels=out_channels, kernel_size=1, )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1_1 = self.gatedconv1_1(x)
        x1_2 = self.gatedconv1_2(x)

        x1 = torch.cat([x1_1, x1_2], dim=1)

        x1 = self.batchnorm1(x1)

        x2_1 = self.gatedconv2_1(x1)
        x2_2 = self.gatedconv2_2(x1)
        x2 = torch.cat([x2_1, x2_2], dim=1)

        x2 = self.batchnorm2(x2)

        x3_1 = self.gatedconv3_1(x2)
        x3_2 = self.gatedconv3_2(x2)
        x3 = torch.cat([x3_1, x3_2], dim=1) + x2

        x3 = self.batchnorm3(x3)
        x3 = self.dropout3(x3)

        x4 = self.gatedconv4(x3)  # + x3

        x4 += x3

        x4 = self.batchnorm4(x4)

        x5_cat = self.gatedconv5(x4)

        x5 = x5_cat + x4

        x5 = self.batchnorm5(x5)

        x6_cat = self.gatedconv6(x5)
        x6 = x6_cat + x5

        x6 = self.batchnorm6(x6)
        x6 = self.dropout6(x6)

        x7_cat = self.gatedconv7(x6)

        x7_cat = self.batchnorm7(x7_cat)

        x8_cat = self.gateconv8(x7_cat)

        x8_cat = self.batchnorm8(x8_cat)

        x_cat = torch.cat([x5_cat, x6_cat, x7_cat, x8_cat], dim=1)

        # x_cat = self.batchnorm8(x_cat)

        x9 = self.conv9(x_cat)

        x10 = self.global_pool(x9)

        return self.softmax(x10)
