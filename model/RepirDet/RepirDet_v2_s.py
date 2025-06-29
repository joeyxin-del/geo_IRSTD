import torch
import torch.nn as nn
# from mmpretrain.models.backbones.mobileone import MobileOne, MobileOneBlock
from .mobiletwo import MobileTwo, MobileOneBlock
from mmdet.models.necks.cspnext_pafpn import CSPNeXtPAFPN
from .lightfpn import FPN
from mmpretrain.models.utils.sparse_modules import (SparseAvgPooling, SparseConv2d, SparseHelper,
                                                    SparseMaxPooling)
from mmpretrain.models.backbones.sparse_convnext import SparseConvNeXtBlock
# from mmpretrain import inference_model
import torch.nn.functional as F
class RepirDet(nn.Module):
    def __init__(self,arch = 'small' ,
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

            MobileOneBlock(
                self.backbone_channels[-3],
                1,
                stride=1,
                kernel_size=3,
                num_convs=4,
                deploy=self.deploy),
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
            out_channels= self.backbone_channels[-4]//2,
            kernel_size=1,
            stride=1
        ),
        nn.BatchNorm2d(self.backbone_channels[-4]//2),
        nn.ReLU(inplace=True)
        )
        self.sparse_head = nn.ModuleList([
            SparseConvNeXtBlock(
                in_channels = self.backbone_channels[-4]//2,
                norm_cfg = dict(type='SparseLN2d', eps=1e-6),
                act_cfg=dict(type='ReLU'),
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
                self.backbone_channels[-4]//2,
                8,
                kernel_size=1,
                stride=1,
                # padding=1
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
            nn.BatchNorm2d(4),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(
                4,
                1,
                kernel_size=3,
                stride=1,
                padding=1
            ),
        )


        self.up2x = nn.Upsample(scale_factor=2,mode='bilinear')
        self.up8x = nn.Upsample(scale_factor=1/2,mode='nearest')
        self.shut2x = nn.Sequential(
            nn.Conv2d(16,out_channels=8,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True)
        )
        self.dense_model_to_sparse(m=self.sparse_head)

        # if mode == 'test':
        #     self.switch_to_deploy()
        if mode == 'train':
            self.backbone.train()



    def forward(self, input):
        ps8x = self.pixel_shuffle_down(self.pixel_shuffle_down(self.pixel_shuffle_down(input)))
        ps16x = self.pixel_shuffle_down(ps8x)
        ps32x = self.pixel_shuffle_down(ps16x)

        # start_time = datetime.now()
        layers, feats2x = self.backbone(input)
        # end_time = datetime.now()
        # print("b：", end_time - start_time)

        feats4x = layers[0]
        # start_time = datetime.now()
        layers[-3] = torch.concat([layers[-3],ps8x], dim=1)
        layers[-2] = torch.concat([layers[-2],ps16x], dim=1)
        layers[-1] = torch.concat([layers[-1],ps32x], dim=1)

        feats = self.neck(layers[-3:])
        # end_time = datetime.now()
        # print("n：", end_time - start_time)
        feats8x, feats16x, feats32x = feats

        # start_time = datetime.now()
        classify_results = self.classify_head(feats8x)
        # end_time = datetime.now()
        # print("c：", end_time - start_time)


        feats8x = self.channel_shut1(torch.concat([feats8x, self.up2x(feats16x)], dim=1))
        feats4x = self.channel_shut2(torch.concat([feats4x, self.up2x(feats8x)], dim=1))
        # feats4x = feats4x + self.channel_shut3(self.up2x(feats8x))
        out = feats4x

        classify_results = F.interpolate(self.classify_head(feats8x),scale_factor=4,mode='bilinear')
        map = self.keep_topk(self.up8x(classify_results).sigmoid())
        SparseHelper._cur_active = map

        # start_time = datetime.now()
        for m in self.sparse_head:
            out = m(out)
        # end_time = datetime.now()
        # print("s：", end_time - start_time)
        out = torch.concat([out, self.shut2x(feats2x)], dim=1)
        out = self.last(out)
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
        with torch.no_grad():
            x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return x

    def keep_topk(self,x,ratio=0.05):
        with torch.no_grad():
            feature_map = x.clone()
            B, _, H, W = feature_map.shape
            # 计算要保留的数据数量（最大的10%）
            num_to_keep = int(ratio * H * W)
            # 展平特征图为一维向量，并复制以进行处理
            flat_feature_map = feature_map.view(B, -1).clone()
            # 对每个Batch中的一维向量进行排序，并找到阈值
            thresholds = torch.topk(flat_feature_map, num_to_keep, dim=1, largest=True).values[:, -1]
            # 使用阈值将每个Batch中的数据分为两部分：大于阈值的保留，小于阈值的置零
            mask = flat_feature_map >= thresholds.view(-1, 1)
            # flat_feature_map = flat_feature_map * mask.float()
            # # 恢复一维向量为原始特征图形状
            # feature_map = flat_feature_map.view(B, 1, H, W)
        return mask.view(B, 1, H, W)



    def switch_to_deploy(self):
        """switch the model to deploy mode, which has smaller amount of
        parameters and calculations."""
        self.backbone.switch_to_deploy()
        for m in self.classify_head:
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
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
    ) as prof:
        out = net(x)
    end_time = datetime.now()
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    # print(out)
    print("程序执行时间为：", end_time - start_time)




