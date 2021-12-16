from torchvision.models import resnet18

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, relu_act=True, batch_norm=False, swish_act=False):
        super(ResidualBlock, self).__init__()
        layers = [nn.ReflectionPad2d(1)]
        layers.append(nn.Conv2d(in_features, in_features, 3))
        if batch_norm:
            layers.append(nn.BatchNorm2d(in_features, 0.8))
        else:
            layers.append(nn.InstanceNorm2d(in_features, affine=True))
        if relu_act:
            if swish_act:
                layers.append(Swish())
            else:
                layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ReflectionPad2d(1))
        layers.append(nn.Conv2d(in_features, in_features, 3))
        if batch_norm:
            layers.append(nn.BatchNorm2d(in_features, 0.8))
        else:
            layers.append(nn.InstanceNorm2d(in_features, affine=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)

class nResNet(nn.Module):
    def __init__(self, num_residual_blocks, out_features, relu_act=True, batch_norm=False, swish_act=False):
        super(nResNet, self).__init__()

        model = [
        ]
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features, relu_act, batch_norm, swish_act)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Swish(nn.Module):
    """Applies the element-wise function :math:`f(x) = x / ( 1 + exp(-x))`
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        # >>> m = nn.Swish()
        # >>> input = autograd.Variable(torch.randn(2))
        # >>> print(input)
        # >>> print(m(input))
    """

    def forward(self, input):
        p = torch.sigmoid(input)
        p = p.mul(input)
        return p

    def __repr__(self):
        return self.__class__.__name__ + ' ()'
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, relu_act=True, swish_act=False,
                 full_style=False):
        super(UNetUp, self).__init__()
        layers = [nn.Upsample(scale_factor=2)]
        layers.append(nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False))
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
        if full_style:
            if relu_act:
                if swish_act:
                    layers.append(Swish())
                else:
                    layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(out_size, out_size, 3, stride=1, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
        if relu_act:
            if swish_act:
                layers.append(Swish())
            else:
                layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout2d(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=False, dropout=0.0, swish_act=False, relu_act=False, style=False):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, stride=2, padding=1, bias=False)]
        if style:
            if normalize:
                layers.append(nn.BatchNorm2d(out_size, 0.8))
            else:
                layers.append(nn.InstanceNorm2d(out_size, affine=True))
            if relu_act:
                if swish_act:
                    layers.append(Swish())
                else:
                    layers.append(nn.LeakyReLU(0.2))

            layers.append(nn.Conv2d(out_size, out_size, 3, stride=1, padding=1))
            if normalize:
                layers.append(nn.BatchNorm2d(out_size, 0.8))
            else:
                layers.append(nn.InstanceNorm2d(out_size, affine=True))
        else:
            if normalize:
                layers.append(nn.BatchNorm2d(out_size, 0.8))
            else:
                layers.append(nn.InstanceNorm2d(out_size, affine=True))
                # else:
        #
        # if relu_act:
        #     if swish_act:
        #         layers.append(Swish())
        #     else:
        #         layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Dropout2d(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
class ResNetUNet(nn.Module):
    def __init__(self,start_channel=3,end_channel=3):
        super(ResNetUNet, self).__init__()
        self.base_model = resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())
        channels = end_channel
        no_resblk = 2
        self.layer1 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer2 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer3 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer4 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer5 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.down1 = UNetDown(start_channel, 64, normalize=False)
        self.down2 = UNetDown(64 + 64, 128)
        self.down3 = UNetDown(128 + 64, 256)
        self.down4 = UNetDown(256 + 128, 512)
        self.down5 = UNetDown(512 + 256, 512)

        self.down6 = UNetDown(512 + 512, 512)
        self.down7 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512)
        self.res1 = nResNet(no_resblk * 4, 512)
        self.up2 = UNetUp(1024, 512)
        self.res2 = nResNet(no_resblk * 4, 512)
        self.up3 = UNetUp(1024, 256)
        self.res3 = nResNet(no_resblk * 4, 512)
        self.up4 = UNetUp(512 + 256, 128)
        self.res4 = nResNet(no_resblk * 4, 256)
        self.up5 = UNetUp(256 + 128, 64)
        self.res5 = nResNet(no_resblk * 4, 128)
        self.up6 = UNetUp(128 + 64, 64)
        self.res6 = nResNet(no_resblk * 4, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(128, channels, (3,3), stride=1, padding=1), nn.Tanh()
        )

    def forward(self, x):
        x = x

        l1 = self.layer1(x)
        d1 = self.down1(x)

        mz = torch.cat((l1, d1), 1)
        d1 = (d1 + l1) / 2
        l2 = self.layer2(l1)
        d2 = self.down2(mz)

        mz = torch.cat((l2, d2), 1)
        l3 = self.layer3(l2)
        d3 = self.down3(mz)

        mz = torch.cat((l3, d3), 1)
        l4 = self.layer4(l3)
        d4 = self.down4(mz)

        mz = torch.cat((l4, d4), 1)
        l5 = self.layer5(l4)
        d5 = self.down5(mz)

        mz = torch.cat((l5, d5), 1)
        d6 = self.down6(mz)

        d7 = self.down7(d6)

        u1 = self.up1(d7, self.res1(d6))
        u2 = self.up2(u1, self.res2(d5))
        u3 = self.up3(u2, self.res3(d4))
        u4 = self.up4(u3, self.res4(d3))
        u5 = self.up5(u4, self.res5(d2))
        u6 = self.up6(u5, self.res6(d1))

        fout = self.final(u6)

        return fout
