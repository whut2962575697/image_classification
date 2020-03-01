




import torch.nn as nn
import torch.nn.functional as F





class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=1,kernel_size=1,padding=0),
            nn.BatchNorm2d(1))
    def forward(self,x):
        x=self.conv(x)
        #print('spatial',x.size())
        x=F.sigmoid(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=int(out_channels/2),kernel_size=1,padding=0),
            nn.BatchNorm2d(int(out_channels/2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels/2),out_channels=out_channels,kernel_size=1,padding=0),
            nn.BatchNorm2d(out_channels)
        )

        self.activation = nn.ReLU()


    def forward(self,x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        #print('channel',x.size())
        x=self.conv1(x)
        x=self.activation(x)
        x=self.conv2(x)
        x=F.sigmoid(x)
        return x


class SCSE_Block(nn.Module):
    def __init__(self, out_channels):
        super(SCSE_Block, self).__init__()
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)

    def forward(self, x):
        g1 = self.spatial_gate(x)
        # print('g1',g1.size())
        g2 = self.channel_gate(x)
        # print('g2',g2.size())
        x = g1 * x + g2 * x
        return x
