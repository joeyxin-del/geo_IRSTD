import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        # x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                                                    size=[height, width],
                                                    mode='bilinear') + x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear')

        return out

class FAMBlock(nn.Module):
    def __init__(self, channels):
        super(FAMBlock, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1
        return out

class DualResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1, planes=16, spp_planes=128, head_planes=128, augment=False):
        super(DualResNet, self).__init__()
        channels = [32, 64, 128, 256]
        # channels = [64, 128, 256, 512]
        # channels = [32//2, 64//2, 128//2, 256//2]
        highres_planes = planes * 8

        self.augment = augment

        self.stem2 = nn.Sequential(
            nn.Conv2d(1, planes, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(),
        )

        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(block, planes, channels[0], layers[0], stride=2)
        # self.layer1 = self._make_layer(block, planes, channels[0], layers[0], stride=1)   # 针对NUDT数据集
        self.layer2 = self._make_layer(block, channels[0], channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[1], channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[2], channels[3], layers[3], stride=2)

        self.FAMBlock3 = FAMBlock(channels=channels[2])
        self.FAMBlock4 = FAMBlock(channels=channels[3])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])
        self.FAM4 = nn.ModuleList([self.FAMBlock4 for i in range(1)])


        self.compression3 = nn.Sequential(   #通道数减少到和 layer2 一样
            nn.Conv2d(channels[2], channels[1], kernel_size=1, bias=False),
            BatchNorm2d(channels[1], momentum=bn_mom),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2d(channels[3], channels[1], kernel_size=1, bias=False),
            BatchNorm2d(channels[1], momentum=bn_mom),
        )

        self.DSB_1 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(channels[2], momentum=bn_mom),
        )

        self.DSB_2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(channels[2], momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(channels[3], momentum=bn_mom),
        )

        # MRF3Net 11月20添加
        self.conv_block1 = nn.Sequential(
            # Vanila_Conv_no_pool(channels[1], channels[0] // 16, 3),
            nn.Conv2d(channels[1], channels[0] // 16, 3, stride=1,padding=1),
            nn.Conv2d(channels[0] // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )

        self.conv_block2 = nn.Sequential(
            # Vanila_Conv_no_pool(channels[1], channels[0] // 16, 3),
            nn.Conv2d(channels[1], channels[0] // 16, 3, stride=1,padding=1),
            nn.Conv2d(channels[0] // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )
        self.modify_multiply = nn.Sequential(
            nn.Conv2d(channels[1], channels[1], 1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU()
            )

        self.modify_sum1 = nn.Sequential(
            nn.Conv2d(channels[2], channels[2], 1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU()
            )
        self.modify_sum2 = nn.Sequential(
            nn.Conv2d(channels[3], channels[3], 1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU()
            )

        # MRF3Net 11月20添加
        # self.dropout = nn.Dropout(0.5)
        self.spp = DAPPM(channels[3], spp_planes, channels[1])

        self.out_plane = nn.Conv2d(channels[1] * 2, channels[1], kernel_size=3, stride=1, padding=1, bias=False)

        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)

        # self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):

        width_output = x.shape[-1] // 4
        height_output = x.shape[-2] // 4

        # width_output = x.shape[-1] // 2     # 针对NUDT数据集
        # height_output = x.shape[-2] // 2    #  针对NUDT数据集

        layers = []

        # x = self.stem(x)    #stem 把通道数提升到planes:16,同时下采样2倍 (B,16,256,256)
        x = self.stem2(x)    #11月,19日修改：stem2 把通道数提升到planes:16,不再下采样2倍 (B,16,512,512)
        # stem = x   # 25年4月11修改

        x = self.layer1(x) # (B,32,512,512)
        layers.append(x)

        x = self.layer2(self.relu(x)) # (B,64,256,256)
        layers.append(x)

        x = self.layer3(self.relu(x)) # (B,128,128,128)
        layers.append(x)

        x_r = F.interpolate(
            self.compression3(self.relu(layers[2])),
            size=[height_output, width_output],
            mode='bilinear')   # (B,128,128,128)
        att_1 = self.conv_block1(x_r)
        x_ = att_1 * layers[1]
        x_ = self.modify_multiply(x_)

        featur_3 = self.DSB_1(x_)  # (B,256,64,64)
        for i in range(2):
            featur_3 = self.FAM3[i](featur_3)  # (B,128,128,128)  CB1
        x_r = x_ + x_r
        x_r_2 = self.modify_multiply(x_r)
        layers[2] = self.modify_sum1(featur_3 + layers[2])  # (B,256,64,64)    DSB 1

        x = self.layer4(layers[2])  # (B,256,64,64)
        layers.append(x)

        x_r = F.interpolate(  # (8,128,64,64)
            self.compression4(self.relu(layers[3])),
            size=[height_output, width_output],
            mode='bilinear')  # (B,128,128,128)

        att_2 = self.conv_block2(x_r)
        x_ = att_2 * x_r_2
        x_ = self.modify_multiply(x_)
        featur_4 = self.DSB_2(x_)  # (B,512,32,32)
        for i in range(1):
            featur_4 = self.FAM4[i](featur_4)
        x_r = x_r + x_  #(8,512,32,32) DSB2
        x_r_3 = self.modify_multiply(x_r)
        layers[3] = self.modify_sum2(layers[3] + featur_4)

        x = x_r_3 + F.interpolate(  # （8,256,64,64）
            self.spp(layers[3]),
            # self.compression4(layers[3]),
            size=[height_output, width_output],
            mode='bilinear')

        x = torch.cat([x, layers[1]], dim=1)
        layers[1] = self.out_plane(x)

        return layers


if __name__ == '__main__':
    '''
        此版本把backbone中的resnet改为 DNANet中的 ResNet-CBAM块,并修剪左边的分支
    # '''
    # model = DualResNet().cuda()
    # x = torch.randn([2,1,512,512]).cuda()
    # out = model(x)
    # print(out)

    from thop import profile
    from torchstat import stat
    from torchsummary import summary

    x = torch.rand(1, 1, 256, 256)
    # model = TADNet()
    model = DualResNet()
    flops, params = profile(model, inputs=(x,))
    print('\n')
    print('Model Params: %2fM || ' % (params / 1e6), end='')
    print('Model FLOPs: %2fGFLOPs' % (flops / 1e9))
    # summary(model, (1, 256, 256))
    # stat(model, (1, 256, 256))  # 使用stat时需要把net.cuda()注释掉或者在cuda上创建测试用的tensor