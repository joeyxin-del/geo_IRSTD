import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TADNet.DDRNet2_12_11_V2 import DualResNet, DAPPM


# from VisualV1 import visual
# outLayerFeature = {}  # 创建全局变量，存储需要可视化层级的featuremap

def Probility_refine(x1, x2):
    w = x1 * x2
    w_sum = x1 * x2 + (1. - x1) * (1. - x2)
    return w / (w_sum + 1e-6)


class CrossLevelFusion(nn.Module):
    def __init__(self, lower_c, higher_c):
        super().__init__()
        # 高层特征处理 - 从higher_c通道降到lower_c通道
        self.higher_conv = nn.Sequential(
            nn.Conv2d(higher_c, lower_c, 1),  # higher_c -> lower_c
            nn.BatchNorm2d(lower_c),
            nn.ReLU()
        )

        self.up2x = nn.Upsample(scale_factor=2, mode='bilinear')

        # 注意力机制 - 使用更小的中间通道数
        mid_c = lower_c // 4  # 降到1/4通道数，与权重文件匹配
        self.att = nn.Sequential(
            nn.Conv2d(2 * lower_c, mid_c, 3, padding=1),  # 2*lower_c -> mid_c
            nn.ReLU(),
            nn.Conv2d(mid_c, 1, 1),
            nn.Sigmoid()
        )

        # 融合后处理 - 使用单个卷积层以匹配权重文件
        self.post_conv = nn.Conv2d(lower_c, lower_c, 3, padding=1)

    def forward(self, lower_feat, higher_feat):
        """
        lower_feat: 低层特征 (lower_c通道)
        higher_feat: 高层特征 (higher_c通道)
        返回: lower_c通道的融合特征
        """
        # 先进行通道对齐
        higher_proj = self.higher_conv(higher_feat)  # higher_c -> lower_c通道
        
        # 再进行尺寸对齐
        higher_proj = self.up2x(higher_proj)  # 保持lower_c通道

        # 生成注意力图
        att_input = torch.cat([lower_feat, higher_proj], dim=1)  # 2*lower_c通道
        att_map = self.att(att_input)  # 1通道

        # 注意力加权融合
        fused = att_map * higher_proj + (1 - att_map) * lower_feat  # lower_c通道
        return self.post_conv(fused)  # lower_c通道


class EnhancedCCM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # Level2处理分支 - 从c2降到c1通道
        self.conv_l2 = nn.Sequential(
            nn.Conv2d(c2, c1, 3, padding=1),  # c2 -> c1通道
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        
        # Level1处理分支
        self.conv_l1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1),  # 保持c1通道不变
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        
        # 注意力机制
        self.att = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, 3, padding=1),  # 2*c1 -> c1
            nn.ReLU(),
            nn.Conv2d(c1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, l1, l2):
        """
        l1: 低层特征 (c1通道)
        l2: 高层特征 (c2通道)
        返回: (enhanced_l1, l2_feat)
        enhanced_l1: c1通道
        l2_feat: c1通道
        """
        # 处理Level2特征，降维到c1通道
        l2_feat = self.conv_l2(l2)  # c2 -> c1通道
        l2_resized = F.interpolate(l2_feat, scale_factor=2, mode='bilinear')  # 保持c1通道

        # 处理Level1特征
        l1_feat = self.conv_l1(l1)  # 保持c1通道

        # 生成注意力图
        att_input = torch.cat([l1_feat, l2_resized], dim=1)  # 2*c1通道
        att_map = self.att(att_input)  # 1通道

        # 残差校准
        enhanced_l1 = l1_feat * att_map + l1_feat  # c1通道
        return enhanced_l1, l2_feat  # 都是c1通道


# class up_conv(nn.Module):
#     """
#     Up Convolution Block
#     """
#
#     def __init__(self, in_ch, out_ch):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         x = self.up(x)
#         return x


# class conv_block(nn.Module):
#     """
#     Convolution Block
#     """
#
#     def __init__(self, in_ch, out_ch):
#         super(conv_block, self).__init__()
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True))
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x


class TADNet(nn.Module):
    def __init__(self, mode="train"):
        super(TADNet, self).__init__()
        self.mode = mode
        self.deep_supervision = True

        self.backbone_channels = [32, 64, 128, 256]  # dualResNet版本

        self.EF3 = EnhancedCCM(self.backbone_channels[2], self.backbone_channels[3])
        self.EF2 = EnhancedCCM(self.backbone_channels[1], self.backbone_channels[2])
        self.EF1 = EnhancedCCM(self.backbone_channels[0], self.backbone_channels[1])

        self.CF3 = CrossLevelFusion(self.backbone_channels[2], self.backbone_channels[3])
        self.CF2 = CrossLevelFusion(self.backbone_channels[1], self.backbone_channels[2])
        self.CF1 = CrossLevelFusion(self.backbone_channels[0], self.backbone_channels[1])

        self.final = nn.Sequential(
            nn.Conv2d(self.backbone_channels[0], self.backbone_channels[0] // 4, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.backbone_channels[0] // 4, 1, 1)
        )

        if self.mode == "train" and self.deep_supervision:
            self.final1 = nn.Sequential(
                nn.Conv2d(self.backbone_channels[0], 1, kernel_size=1),
                nn.Upsample(scale_factor=2)
            )
            self.final2 = nn.Sequential(
                nn.Conv2d(self.backbone_channels[1], 1, kernel_size=1),
                nn.Upsample(scale_factor=4)
            )

        self.down_dualResNet = DualResNet(layers=[2, 2, 2, 2])

    def forward(self, x):
        x = self.down_dualResNet(x)
        x1, x2, x3, x4 = x  # 32, 64, 128, 256 channels

        # EF3: 处理x3(128)和x4(256)
        o3, o4 = self.EF3(x3, x4)  # o3(128), o4(128)
        o3 = self.CF3(o3, x4)  # o3(128), x4(256) -> o3(128)

        # EF2: 处理x2(64)和o3(128)
        o2, o3_new = self.EF2(x2, o3)  # o2(64), o3_new(64)
        o2 = self.CF2(o2, o3)  # o2(64), o3(128) -> o2(64)

        # EF1: 处理x1(32)和o2(64)
        o1, o2_new = self.EF1(x1, o2)  # o1(32), o2_new(32)
        o1 = self.CF1(o1, o2)  # o1(32), o2(64) -> o1(32)

        x0 = self.final(o1)
        out = torch.sigmoid(x0)

        if self.deep_supervision:
            output1 = self.final1(x1).sigmoid()
            output2 = self.final2(x2).sigmoid()
            return [out, output1, output2]
        else:
            return out


if __name__ == '__main__':
    from thop import profile
    from torchstat import stat
    from torchsummary import summary

    # x = torch.rand((3,1,256,256))
    # x = torch.rand((2, 1, 512, 512))
    # x = x.to('cuda')
    # model = TADNet()
    # model.to('cuda')
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Total_params: {}".format(pytorch_total_params))
    # out = model(x)
    # print(out[0])

    from thop import profile
    x = torch.rand((1, 1, 256, 256))
    model = TADNet()
    # out = model(x)
    # # print(out.size)

    flops, params = profile(model, inputs=(x,))
    print('\n')
    print('Model Params: %2fM || ' % (params / 1e6), end='')
    print('Model FLOPs: %2fGFLOPs' % (flops / 1e9))

    # summary(model, (1, 512, 512))

    # stat(model, (1, 256, 256))  # 使用stat时需要把net.cuda()注释掉或者在cuda上创建测试用的tensor