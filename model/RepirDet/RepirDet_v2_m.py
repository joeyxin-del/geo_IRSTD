import torch
import torch.nn as nn
# from mmpretrain.models.backbones.mobileone import MobileOne, MobileOneBlock
from .mobiletwo import MobileTwo, MobileOneBlock
from mmdet.models.necks.cspnext_pafpn import CSPNeXtPAFPN
from mmdet.models.necks.fpn import FPN
from mmpretrain.models.utils.sparse_modules import (SparseAvgPooling, SparseConv2d, SparseHelper,
                                                    SparseMaxPooling)
from mmpretrain.models.backbones.sparse_convnext import SparseConvNeXtBlock
# from mmpretrain import inference_model
import torch.nn.functional as F
from torch.nn.modules.utils import _triple, _pair, _single
import torchvision.utils as vutils
# from VisualV1 import visual



outLayerFeature = {}  #创建全局变量，存储需要可视化层级的Featuremap
class Competition(nn.Module):
    def __init__(self,in_channels,kernel_size=9):
        super(Competition, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels,in_channels*2,1),
                                   nn.BatchNorm2d(in_channels*2),
                                   nn.ReLU6())
        self.kernel_size = kernel_size
        # self.conv2 = nn.Conv2d(in_channels,1,self.kernel_size,1,padding = self.kernel_size // 2)
        # self.up = nn.PixelShuffle(2)
        self.channs = in_channels

    def forward(self,x):
        x = self.conv1(x)
        x1 = x[:,0:self.channs,:,:]
        x2 = x[:,self.channs:,:,:]
        # x2 = self.conv2(x2).repeat([1,self.channs,1,1]) #resourse

        e_x = torch.exp(x1)
        kernel_size = self.kernel_size
        return F.avg_pool2d(x2.mul(e_x), kernel_size, stride=1, padding = kernel_size // 2).mul_(kernel_size**2).div_(
            F.avg_pool2d(e_x, kernel_size, stride=1, padding = kernel_size // 2).mul_(kernel_size**2))

class Competition2(nn.Module):
    def __init__(self):
        super(Competition2, self).__init__()
        self.down = nn.PixelUnshuffle(downscale_factor=8)
        self.up = nn.PixelShuffle(upscale_factor=8)

    def forward(self,x):
        x = self.down(x) **2
        B,C,H,W = x.shape
        x = x.reshape([B, C, -1]).softmax(dim=-1).reshape(B,C,H,W) * x
        x = self.up(x)
        return x


class RepirDet(nn.Module):
    def __init__(self,arch = 'light' ,
                 deploy=False,
                 mode='train'):
        super(RepirDet, self).__init__()


        """
            hyper parameter
        """
        self.backbone_channels = [64//2, 128//2, 256//2, 512//2]
        """"""
        self.backbone_channels_cat = [64//2 + 0, 128//2 + 64, 256//2 + 64*4, 512//2 + 64*16]

        self.deploy = deploy

        self.backbone = MobileTwo(
            arch=arch,
            in_channels=1,
            out_indices=(0, 1, 2, 3),
            deploy = self.deploy,
        )

        self.neck = FPN(
            in_channels=self.backbone_channels_cat[-3:],
            out_channels=self.backbone_channels[-3],
            # use_depthwise=True,
            act_cfg=dict(type='ReLU'),
            num_outs=len(self.backbone_channels[-3:])
            # use_depthwise=True,
        )
        self.classify_head = nn.Sequential(
            MobileOneBlock(
            self.backbone_channels[-3],
            self.backbone_channels[-3],
            stride=1,
            kernel_size=3,
            num_convs=4,
            deploy=self.deploy),
            nn.Conv2d(
                in_channels = self.backbone_channels[-3],
                out_channels = 1,
                kernel_size = 1,
                stride=1
            ),
        #     MobileOneBlock(
        #         self.backbone_channels[-3],
        #         1,
        #         stride=1,
        #         kernel_size=3,
        #         num_convs=4,
        #         deploy=self.deploy),
        )
        self.channel_shut1 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.backbone_channels[-3] * 2,
                out_channels = self.backbone_channels[-3],
                kernel_size = 1,
                stride=1
            ),
            nn.BatchNorm2d(self.backbone_channels[-3]),
            nn.ReLU(inplace=True)
        )
        self.channel_shut2 = nn.Sequential(nn.Conv2d(
            in_channels=self.backbone_channels[-3] + self.backbone_channels[-4],
            out_channels= self.backbone_channels[-4],
            kernel_size=1,
            stride=1
        ),
        nn.BatchNorm2d(self.backbone_channels[-4]),
        nn.ReLU(inplace=True)
        )
        self.sparse_head = nn.ModuleList([
            SparseConvNeXtBlock(
                in_channels = self.backbone_channels[-4],
                norm_cfg = dict(type='SparseLN2d', eps=1e-6),
                act_cfg=dict(type='GELU'),
                linear_pw_conv = True,
                layer_scale_init_value = 0.,
                mlp_ratio=1.,
                use_grn=True,
                with_cp=False
            ) for _ in range(1)],
        )
        # self.sparse_head.append(
        #     nn.Conv2d(
        #         self.backbone_channels[-4]*2,
        #         16,
        #         kernel_size=1,
        #         stride=1,
        #     )
        # )
        self.upsample = nn.Sequential(
            nn.Conv2d(
                self.backbone_channels[-4],
                8,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(8),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            )
        self.sparse_head.append(self.upsample)

        self.last = nn.Sequential(
            nn.Conv2d(
                8*2,
                4,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.last2 = nn.Sequential(

            nn.Conv2d(
                4,
                1,
                kernel_size=5,
                stride=1,
                padding=5//2
            ),
        )


        self.up2x = nn.Upsample(scale_factor=2,mode='bilinear')
        self.up8x = nn.Upsample(scale_factor=1/2,mode='nearest')
        self.shut2x = nn.Sequential(
            # # 新增
            # nn.Conv2d(16,out_channels=16,kernel_size=3,stride=1,padding=1),
            # nn.BatchNorm2d(num_features=16),
            # nn.ReLU(inplace=True),

            nn.Conv2d(16,out_channels=8,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True)
        )
        # self.competition = Competition(in_channels=8*2)
        self.competition = Competition2()

        self.idx_iter=-1

        # self.dense_model_to_sparse(m=self.sparse_head)
        # self.dense_model_to_sparse(m=self.last2)

        # if mode == 'test':
        #     self.switch_to_deploy()
        if mode == 'train':
            self.backbone.train()



    def forward(self, input): # (B, 1, 512, 512)
        ps8x = self.pixel_shuffle_down(self.pixel_shuffle_down(self.pixel_shuffle_down(input))) #(B,64,64,64)
        ps16x = self.pixel_shuffle_down(ps8x) #(B,256,32,32)
        ps32x = self.pixel_shuffle_down(ps16x) #(B,1024,16,16)

        # outLayerFeature["ps8x"] = ps8x
        # outLayerFeature["ps16x"] = ps16x
        # outLayerFeature["ps32x"] = ps32x

        # start_time = datetime.now()

        # outLayerFeature["0_input"] = input
        # visual(None, outLayerFeature)

        layers, feats2x = self.backbone(input) # layers{[b,32,128,128],[b,64,64,64],[b,128,32,32],[b,256,16,16]}, feats2x(b,16,256,256)
        # end_time = datetime.now()
        # print("b：", end_time - start_time)


        # outLayerFeature["0.input"] = input
        # outLayerFeature["1.backbone_feats2x"] = feats2x
        # outLayerFeature["2.backbone_feats4X"] = layers[0]
        # outLayerFeature["2.backbone_feats8X"] = layers[1]
        # outLayerFeature["2.backbone_feats16X"] = layers[2]
        # outLayerFeature["2.backbone_feats32X"] = layers[3]

        feats4x = layers[0]   #(b,32,128,128)
        # start_time = datetime.now()
        layers[-3] = torch.concat([layers[-3],ps8x], dim=1) #(B,128,64,64)
        layers[-2] = torch.concat([layers[-2],ps16x], dim=1) #(B,384,32,32)
        layers[-1] = torch.concat([layers[-1],ps32x], dim=1) #(B,1280,16,16)

        # outLayerFeature["3.before_neck_feats8x"] = layers[-3]
        # outLayerFeature["3.before_neck_feats16x"] = layers[-2]
        # outLayerFeature["3.before_neck_feats32x"] = layers[-1]

        feats = self.neck(layers[-3:])    # FPN (b,64,64,64),(b,64,32,32),(b,64,16,16)

        # end_time = datetime.now()
        # print("n：", end_time - start_time)
        feats8x, feats16x, feats32x = feats #(B,64,64,64),(B,64,32,32),(B,64,16,16)
        # outLayerFeature["3.after_neck_feats8x"] = feats8x
        # outLayerFeature["3.after_neck_feats16x"] = feats16x
        # outLayerFeature["3.after_neck_feats32x"] = feats32x




        # start_time = datetime.now()
        classify_results = F.interpolate(self.classify_head(feats8x),scale_factor=4,mode='bilinear') #(B,1,256,256)
        # end_time = datetime.now()
        # print("c：", end_time - start_time)
        # outLayerFeature["4.classify_results_feats8x"] = classify_results



        feats8x = self.channel_shut1(torch.concat([feats8x, self.up2x(feats16x)], dim=1)) #(B,64,64,64)
        feats4x = self.channel_shut2(torch.concat([feats4x, self.up2x(feats8x)], dim=1))  #(B,32,128,128)
        # feats4x = feats4x + self.channel_shut3(self.up2x(feats8x))

        # outLayerFeature["5.channel_shut1_feats8x"] = feats8x
        # outLayerFeature["5.channel_shut2_feats4x"] = feats4x

        out = feats4x #(B,32,128,128)

        map = self.keep_topk(self.up8x(classify_results).sigmoid()) #(B,1,128,128)

        # outLayerFeature["6.keep_topk_classify_results_feats8x"] = map

        SparseHelper._cur_active = map



        # self.idx_iter = (self.idx_iter+1) % 50
        # vutils.save_image(classify_results.sigmoid(), f"K:\\BasicIRSTD-main\\output\\class_{(self.idx_iter%80)}.png")
        # vutils.save_image(map.float(), f"K:\\BasicIRSTD-main\\output\\map_{(self.idx_iter%80)}.png")

        # start_time = datetime.now()
        for m in self.sparse_head:
            out = m(out)   #(B,32,128,128)-->(B,32,128,128)-->(B,8,256,256)
        # end_time = datetime.now()
        # print("s：", end_time - start_time)
        # outLayerFeature["6.sparse_head_4x"] = out

        out = torch.concat([out, self.shut2x(feats2x)], dim=1) #(B,16,256,256)

        # #新增
        # B,C,H,W = out.shape
        # out = out.reshape([B, C, -1]).softmax(dim=-1).reshape(B,C,H,W) * out
        # 另一个竞争机制
        # out = self.competition(out)

        out = self.last2(self.last(out)) #(B,1,512,512)
        # outLayerFeature["7.last2.sigmoid"] = out.sigmoid()
        # outLayerFeature["8.classify_results.sigmoid"] = classify_results.sigmoid()
        # visual(input, outLayerFeature)
        return [out.sigmoid(), classify_results.sigmoid()]

    def dense_model_to_sparse(self, m: nn.Module) -> nn.Module:
        """Convert regular dense modules to sparse modules."""
        output = m
        if isinstance(m, nn.Conv2d):
            m: nn.Conv2d
            bias = m.bias is not None
            output = SparseConv2d(
                m.in_channels,
                m.out_channels,
                kernel_size=m.kernel_size,
                stride=m.stride,
                padding=m.padding,
                dilation=m.dilation,
                groups=m.groups,
                bias=bias,
                padding_mode=m.padding_mode,
            )
            output.weight.data.copy_(m.weight.data)
            if bias:
                output.bias.data.copy_(m.bias.data)

        elif isinstance(m, nn.MaxPool2d):
            m: nn.MaxPool2d
            output = SparseMaxPooling(
                m.kernel_size,
                stride=m.stride,
                padding=m.padding,
                dilation=m.dilation,
                return_indices=m.return_indices,
                ceil_mode=m.ceil_mode)

        elif isinstance(m, nn.AvgPool2d):
            m: nn.AvgPool2d
            output = SparseAvgPooling(
                m.kernel_size,
                m.stride,
                m.padding,
                ceil_mode=m.ceil_mode,
                count_include_pad=m.count_include_pad,
                divisor_override=m.divisor_override)

        # elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        #     m: nn.BatchNorm2d
        #     output = (SparseSyncBatchNorm2d
        #               if enable_sync_bn else SparseBatchNorm2d)(
        #                   m.weight.shape[0],
        #                   eps=m.eps,
        #                   momentum=m.momentum,
        #                   affine=m.affine,
        #                   track_running_stats=m.track_running_stats)
        #     output.weight.data.copy_(m.weight.data)
        #     output.bias.data.copy_(m.bias.data)
        #     output.running_mean.data.copy_(m.running_mean.data)
        #     output.running_var.data.copy_(m.running_var.data)
        #     output.num_batches_tracked.data.copy_(m.num_batches_tracked.data)

        for name, child in m.named_children():
            output.add_module(name, self.dense_model_to_sparse(child))
        del m
        return output

    def pixel_shuffle_down(self,x):
        # with torch.no_grad():
            # x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = torch.pixel_unshuffle(x,2)
        return x

    def keep_topk(self,x,ratio=0.05):
        feature_map = x.clone()
        B, _, H, W = feature_map.shape
            # 计算要保留的数据数量（最大的10%）
        num_to_keep = int(ratio * H * W)
            # 展平特征图为一维向量，并复制以进行处理
        flat_feature_map = feature_map.view(B, -1).clone()
            # 对每个Batch中的一维向量进行排序，并找到阈值
        thresholds = torch.topk(flat_feature_map, num_to_keep, dim=1, largest=True).values[:, -1]
            # 使用阈值将每个Batch中的数据分为两部分：大于阈值的保留，小于阈值的置零
        mask = flat_feature_map > thresholds.view(-1, 1)
            # flat_feature_map = flat_feature_map * mask.float()
            # # 恢复一维向量为原始特征图形状
            # feature_map = flat_feature_map.view(B, 1, H, W)
        return mask.view(B, 1, H, W)



    def switch_to_deploy(self):
        """switch the model to deploy mode, which has smaller amount of
        parameters and calculations."""
        self.backbone.switch_to_deploy()
        for m in self.classify_head:
            if isinstance(m,MobileOneBlock):
                m.switch_to_deploy()
        for m in self.sparse_head:
            if isinstance(m,MobileOneBlock):
                m.switch_to_deploy()
        # self.deploy = True



from datetime import datetime
if __name__ == '__main__':
    x = torch.randn(8,1,512,512)
    net= RepirDet(deploy=True)

    start_time = datetime.now()
    out = net(x)
    end_time = datetime.now()
    # print(out)
    print("程序执行时间为：", end_time - start_time)




