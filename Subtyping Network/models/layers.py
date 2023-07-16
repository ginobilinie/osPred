import torch.nn as nn


class ConvBlock(nn.Module):

    #Specific convolutional block followed by batch normalization and leakyrelu for unet

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        #self.dropout = nn.Dropout(p=0.5)##
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.main(x)
        out = self.bn(out)
        #out = self.dropout(out)
        out = self.activation(out)##
        return out

