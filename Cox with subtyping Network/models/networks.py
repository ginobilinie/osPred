import torch
import torch.nn as nn
import random
from models.layers import conv3x3, ConvBlock


def mixup_data(x, y, lam, use_cuda=True):
    ###Returns mixed inputs, pairs of targets, and lambda
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b


class Classifier(nn.Module):
    def __init__(self, in_planes, nf_enc):
        super().__init__()

        self.encoder = nn.ModuleList()  # G1. -> G6.

        prev_nf = in_planes# 7->16->32->64->64+30->256->64->3
                           # 0  1   2   3          4    5
        for nf in nf_enc:
            self.encoder.append(
                nn.Sequential(
                    ConvBlock(prev_nf, nf),
                    nn.MaxPool3d(2, 2)
                )
            )
            prev_nf = nf

        #channels = nf+3*32
        self.last_conv = nn.Sequential(
            ConvBlock(nf, nf),
            nn.AdaptiveAvgPool3d(1)
        )

        self.fc = nn.Linear(nf+1, 256) # G8.
        self.dropout1 = nn.Dropout(p=0.5)##

        self.tri_pathway = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.Dropout(p=0.5),##
                nn.ReLU()
            ),
            nn.Linear(64, 3)
        ])


    def forward(self, x, info, label=None, lam=0.2, mixup_hidden=False, layer_mix=None):


        if mixup_hidden == True:
            if layer_mix == None:
                layer_mix = random.randint(0,6)
            if layer_mix == 0:
                x, y1, y2 =mixup_data(x,label,lam)
                x = self.encoder[0](x)#7->16
                x = self.encoder[1](x)#16->32
                x = self.encoder[2](x)#32->64
                x = self.last_conv(x)#64->64
                x = torch.flatten(x, start_dim=1)#64 64
                x = torch.cat([x, info*lam+info[y2]*(1-lam)], dim=1)
                x = self.fc(x) # ->256
                x = self.dropout1(x)
                x = self.tri_pathway[0](x) #256->64
                out_tri = self.tri_pathway[1](x) #64->3
                return out_tri,y1,y2

            if layer_mix == 1:
                x = self.encoder[0](x)  # 7->16
                x, y1, y2 = mixup_data(x, label, lam)
                x = self.encoder[1](x)  # 16->32
                x = self.encoder[2](x)  # 32->64
                x = self.last_conv(x)  # 64->64
                x = torch.flatten(x, start_dim=1)  # 64 64
                x = torch.cat([x, info * lam + info[y2] * (1 - lam)], dim=1)
                x = self.fc(x)  # ->256
                x = self.dropout1(x)
                x = self.tri_pathway[0](x)  # 256->64
                out_tri = self.tri_pathway[1](x)  # 64->3
                return out_tri, y1, y2
            if layer_mix == 2:
                x = self.encoder[0](x)  # 7->16
                x = self.encoder[1](x)  # 16->32
                x, y1, y2 = mixup_data(x, label, lam)
                x = self.encoder[2](x)  # 32->64
                x = self.last_conv(x)  # 64->64
                x = torch.flatten(x, start_dim=1)  # 64 64
                x = torch.cat([x, info * lam + info[y2] * (1 - lam)], dim=1)
                x = self.fc(x)  # ->256
                x = self.dropout1(x)
                x = self.tri_pathway[0](x)  # 256->64
                out_tri = self.tri_pathway[1](x)  # 64->3

                return out_tri, y1, y2
            if layer_mix == 3:
                x = self.encoder[0](x)  # 7->16
                x = self.encoder[1](x)  # 16->32
                x = self.encoder[2](x)  # 32->64
                x, y1, y2 = mixup_data(x, label, lam)
                x = self.last_conv(x)  # 64->64
                x = torch.flatten(x, start_dim=1)  # 64 64
                x = torch.cat([x, info * lam + info[y2] * (1 - lam)], dim=1)
                x = self.fc(x)  # ->256
                x = self.dropout1(x)

                x = self.tri_pathway[0](x)  # 256->64
                out_tri = self.tri_pathway[1](x)  # 64->3

                return out_tri, y1, y2
            if layer_mix == 4:
                x = self.encoder[0](x)  # 7->16
                x = self.encoder[1](x)  # 16->32
                x = self.encoder[2](x)  # 32->64
                x = self.last_conv(x)  # 64->64
                x, y1, y2 = mixup_data(x, label, lam)
                x = torch.flatten(x, start_dim=1)  # 64 64
                x = torch.cat([x, info * lam + info[y2] * (1 - lam)], dim=1)
                x = self.fc(x)  # ->256
                x = self.dropout1(x)
                x = self.tri_pathway[0](x)  # 256->64
                out_tri = self.tri_pathway[1](x)  # 64->3

                return out_tri, y1, y2
            if layer_mix == 5:
                x = self.encoder[0](x)  # 7->16
                x = self.encoder[1](x)  # 16->32
                x = self.encoder[2](x)  # 32->64
                x = self.last_conv(x)  # 64->64
                x = torch.flatten(x, start_dim=1)  # 64 64
                x = torch.cat([x, info], dim=1)
                x = self.fc(x)  # ->256
                x, y1, y2 = mixup_data(x, label, lam)
                x = self.dropout1(x)

                x = self.tri_pathway[0](x)  # 256->64
                out_tri = self.tri_pathway[1](x)  # 64->3

                return out_tri, y1, y2
            if layer_mix == 6:
                x = self.encoder[0](x)  # 7->16
                x = self.encoder[1](x)  # 16->32
                x = self.encoder[2](x)  # 32->64
                x = self.last_conv(x)  # 64->64
                x = torch.flatten(x, start_dim=1)  # 64 64
                x = torch.cat([x, info], dim=1)
                x = self.fc(x)  # ->256
                x = self.dropout1(x)

                x = self.tri_pathway[0](x)  # 256->64
                x, y1, y2 = mixup_data(x, label, lam)
                out_tri = self.tri_pathway[1](x)  # 64->3

                return out_tri, y1, y2
        else:

            x = self.encoder[0](x)  # 7->16
            x = self.encoder[1](x)  # 16->32
            x = self.encoder[2](x)  # 32->64
            x = self.last_conv(x)
            x = torch.flatten(x, start_dim=1)
            x = torch.cat([x, info], dim=1)
            x = self.fc(x)
            x = self.tri_pathway[0](x)  # 256->64

            out_tri = self.tri_pathway[1](x)
            return x


class OSNet(nn.Module):
    """
    生存时间预测网络
    """

    def __init__(self, in_channels, model1,  conv_nf=[16, 32, 64]):
        super().__init__()

        self.model1 = model1
        self.model = nn.ModuleList()
        prev_nf = in_channels
        for nf in conv_nf:
            self.model.append(
                nn.Sequential(
                    conv3x3(prev_nf, nf, dropout=0.0),
                    nn.MaxPool3d(2)
                )
            )
            prev_nf = nf

        self.last_conv = conv3x3(nf, nf, dropout=0.0)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.out = nn.Linear(nf + 65, 1)

        self.grad = {}

    def save_grad(self, name):
        def hook(grad):
            self.grad[name] = grad

        return hook

    def forward(self, x, info):
        data = x
        for layer in self.model:
            x = layer(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).flatten(start_dim=1)
        x1 = self.model1(data, info)
        x = torch.cat([x, x1], dim=1)
        x = torch.cat([x, info], dim=1)
        x = self.out(x)

        return x

