from typing import Dict, List, Tuple, Union, Optional, Type, Callable, Any
from inspect import signature
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from einops import rearrange
# from .softpool import SoftPool2d

__all__ = [
    "efficientvit_b0",
    "efficientvit_b1",
    "efficientvit_b2",
    "efficientvit_b3",
]


#################################################################################
#                             Basic Layers                                      #
#################################################################################

def build_kwargs_from_config(config: Dict, target_func: Callable) -> Dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


REGISTERED_NORM_DICT: Dict[str, Type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name == "ln":
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None

class ExpActivation(nn.Module):
    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, input):
        out = torch.exp(input/10)
        return out
class Eluplus1(nn.Module):
    def __init__(self):
        super(Eluplus1, self).__init__()

    def forward(self, input):
        out = F.elu(input)+1.
        return out

REGISTERED_ACT_DICT: Dict[str, Type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "sigmoid": nn.Sigmoid,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "exp":ExpActivation,
    "elup1": Eluplus1,
}


def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None


def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def list_sum(x: List) -> Any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def merge_tensor(x: List[torch.Tensor], mode="cat", dim=1) -> torch.Tensor:
    if mode == "cat":
        return torch.cat(x, dim=dim)
    elif mode == "add":
        return list_sum(x)
    else:
        raise NotImplementedError


def resize(
        x: torch.Tensor,
        size: Optional[Any] = None,
        scale_factor: Optional[List[float]] = None,
        mode: str = "bicubic",
        align_corners: Optional[bool] = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


def val2list(x: Union[List, Tuple, Any], repeat_time=1) -> List:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: Union[List, Tuple, Any], min_len: int = 1, idx_repeat: int = -1) -> Tuple:
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def bhwc2bchw( x):
    # (B, H * W, C) --> (B, C, H, W)
    B,S,HW, C = x.shape
    H = int(HW ** (0.5))
    x = x.reshape(-1, H, H, C).permute(0, 3, 1, 2)
    return x,[B,S,HW, C]

def bchw2bhwc( x,S):
    x = x.reshape(S[0], S[1],S[3], S[2]).permute(0,1, 3, 2)
    return x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm

def img2windowsCHW(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

def windows2imgCHW(img_splits_hw,B, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' C H W
    """

    img = img_splits_hw.reshape(B, -1, H // H_sp, W // W_sp, H_sp, W_sp)
    img = img.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    return img


class ConvLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=1,
            padding=None,
            use_bias=False,
            dropout_rate=0.2,
            norm="bn2d",
            act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.dropout = nn.Dropout2d(dropout_rate, inplace=False) if dropout_rate > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    def __init__(
            self,
            mode="bicubic",
            size: Union[int, Tuple[int, int], List[int], None] = None,
            factor=2,
            align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class LinearLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            use_bias=True,
            dropout_rate=0.2,
            norm=None,
            act_func=None,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout_rate, inplace=False) if dropout_rate > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x



#################################################################################
#                             Basic Blocks                                      #
#################################################################################


class DSConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
            use_bias=False,
            norm=("bn2d", "bn2d"),
            act_func=("relu6", None),
    ):
        # 初始化DSConv类，包含深度卷积和点卷积
        super(DSConv, self).__init__()

        # 将参数转换为元组形式，确保支持不同维度的参数配置
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        # 定义深度卷积层：对每个通道分别做卷积，通道数保持不变
        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,  # 每个输入通道独立卷积（深度卷积）
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )

        # 定义点卷积层：使用1x1卷积改变通道数
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播：先经过深度卷积，再经过点卷积
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
            mid_channels=None,
            expand_ratio=6,
            use_bias=False,
            norm=("bn2d", "bn2d", "bn2d"),
            act_func=("relu6", "relu6", None),
    ):
        # 初始化MBConv类，包含反向瓶颈卷积、深度卷积和点卷积
        super(MBConv, self).__init__()

        # 将参数转换为元组形式
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)

        # mid_channels: 计算中间层通道数，默认通过in_channels和expand_ratio决定
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        # 定义反向瓶颈卷积：1x1卷积将输入通道数扩展为mid_channels
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )

        # 定义深度卷积：深度可分离卷积
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,  # 深度可分离卷积
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

        # 定义点卷积：使用1x1卷积改变通道数
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播：先经过反向瓶颈卷积，再经过深度卷积，最后经过点卷积
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class LiteMSA(nn.Module):
    r""" Lightweight multi-scale attention """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: Optional[int] = None,
            heads_ratio: float = 1.0,
            dim=8,
            use_bias=False,
            norm=(None, "bn2d"),
            act_func=(None, None),
            kernel_func="relu",
            scales: Tuple[int, ...] = (5,),
    ):
        # 初始化LiteMSA类，表示轻量级多尺度注意力机制
        super(LiteMSA, self).__init__()

        # 计算heads数量，若未传入则根据输入通道数和dim计算
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim  # 每个头的维度总和

        # 将参数转换为元组形式
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        # 定义qkv（查询、键、值）卷积层，将输入映射到查询、键、值的维度
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )

        # 定义多尺度聚合层：使用不同尺度的卷积操作来生成多尺度特征
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim, 3 * total_dim, scale, padding=get_same_padding(scale), groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),

                    nn.Conv2d(
                        3 * total_dim, 3 * total_dim, 1, groups=3 * total_dim,
                        bias=use_bias[0],),
                )
                for scale in scales  # 为每个尺度定义卷积层
            ]
        )

        # 定义核函数，通常用于激活函数
        self.kernel_func = build_act(kernel_func, inplace=False)

        # 定义最终投影层，将多尺度特征合并后映射到输出通道
        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())  # 获取输入的批大小、通道数、高度和宽度

        # 生成多尺度的q, k, v（查询、键、值）
        # print("x shape:", x.shape)
        qkv = self.qkv(x)  # 得到qkv的初步映射
        # print("qkv shape:", qkv.shape)
        multi_scale_qkv = [qkv]  # 存储多尺度qkv
        for op in self.aggreg:  # 针对不同尺度的聚合
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)  # 将不同尺度的qkv拼接起来
        # print("multi_scale_qkv shape:", multi_scale_qkv.shape)
        # 重塑多尺度qkv的形状
        multi_scale_qkv = torch.reshape(
            multi_scale_qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        # print("multi_scale_qkv shape:", multi_scale_qkv.shape)
        multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)  # 转置qkv
        # print("multi_scale_qkv shape:", multi_scale_qkv.shape)
        # 分离q, k, v
        q, k, v = (
            multi_scale_qkv[..., 0: self.dim].clone(),
            multi_scale_qkv[..., self.dim: 2 * self.dim].clone(),
            multi_scale_qkv[..., 2 * self.dim:].clone(),
        )
        # print("multi_scale_qkv shape:", multi_scale_qkv.shape)
        # 对q, k应用激活函数
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        # print("q shape:", q.shape)
        # print("k shape:", k.shape)
        # 对v进行转置
        trans_k = k.transpose(-1, -2)

        # 对v进行padding，并计算kv矩阵
        v = F.pad(v, (0, 1), mode="constant", value=1)
        # print("v shape:", v.shape)
        kv = torch.matmul(trans_k, v)
        # print("kv shape:", kv.shape)
        out = torch.matmul(q, kv)  # 计算注意力输出
        # print("out shape:", out.shape)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)  # 最后一个维度归一化
        # print("out shape:", out.shape)
        # 转置并重塑输出，最后进行投影映射
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        out = self.proj(out)
        # print("out shape:", out.shape)
        return out


class EfficientViTBlock(nn.Module):
    def __init__(self, in_channels: int, heads_ratio: float = 1.0, dim=32, expand_ratio: float = 4, norm="bn2d",
                 act_func="relu"):  # hswiash
        # 初始化EfficientViTBlock类，继承自nn.Module，定义了模型的各个模块
        super(EfficientViTBlock, self).__init__()
        # 调用父类的初始化方法，初始化模块

        # 初始化context_module，包含了LiteMSA模块和ResidualBlock模块
        self.context_module = ResidualBlock(
            LiteMSA(
                in_channels=in_channels,  # 输入通道数
                out_channels=in_channels,  # 输出通道数保持不变
                heads_ratio=heads_ratio,  # 多头注意力的头部比例
                dim=dim,  # 每个头的维度
                heads=4,  # 多头注意力的头数
                norm=(None, norm),  # 正则化类型, 第一个None表示LiteMSA中的正则化，第二个是用于norm的类型
            ),
            IdentityLayer(),  # IdentityLayer作为残差连接的第二部分
        )

        # 初始化local_module，包含了MBConv模块和ResidualBlock模块
        local_module = MBConv(
            in_channels=in_channels,  # 输入通道数
            out_channels=in_channels,  # 输出通道数保持不变
            expand_ratio=expand_ratio,  # 通道扩展比例
            use_bias=(False, False, False),  # 使用偏置的设置
            norm=(None, None, norm),  # 规范化类型, None表示在不同部分的规范化类型
            act_func=(act_func, act_func, None),  # 激活函数设置，前两部分使用给定的激活函数，最后一部分没有激活函数
        )

        # 使用ResidualBlock来包裹local_module，并作为local_module模块
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 定义前向传播过程
        x = self.context_module(x)  # 先通过context_module（包含LiteMSA和ResidualBlock）
        x = self.local_module(x)  # 再通过local_module（包含MBConv和ResidualBlock）
        return x  # 返回结果

class EfficientViTBlock_downsample(nn.Module):
    def __init__(self, in_channels: int, heads_ratio: float = 1.0, dim=32, expand_ratio: float = 4, norm="bn2d",
                 act_func="relu"):  # hswiash
        # 初始化EfficientViTBlock类，继承自nn.Module，定义了模型的各个模块
        super(EfficientViTBlock_downsample, self).__init__()
        # 调用父类的初始化方法，初始化模块

        # 初始化context_module，包含了LiteMSA模块和ResidualBlock模块
        self.context_module = ResidualBlock(
            LiteMSA(
                in_channels=in_channels,  # 输入通道数
                out_channels=in_channels,  # 输出通道数保持不变
                heads_ratio=heads_ratio,  # 多头注意力的头部比例
                dim=dim,  # 每个头的维度
                heads=4,  # 多头注意力的头数
                norm=(None, norm),  # 正则化类型, 第一个None表示LiteMSA中的正则化，第二个是用于norm的类型
            ),
            IdentityLayer(),  # IdentityLayer作为残差连接的第二部分
        )

        # 初始化local_module，包含了MBConv模块和ResidualBlock模块
        local_module = MBConv(
            in_channels=in_channels,  # 输入通道数
            out_channels=in_channels*2,  # 输出通道数保持不变
            expand_ratio=expand_ratio,  # 通道扩展比例
            use_bias=(False, False, False),  # 使用偏置的设置
            stride=2,
            norm=(None, None, norm),  # 规范化类型, None表示在不同部分的规范化类型
            act_func=(act_func, act_func, None),  # 激活函数设置，前两部分使用给定的激活函数，最后一部分没有激活函数
        )

        # 使用ResidualBlock来包裹local_module，并作为local_module模块
        # self.local_module = ResidualBlock(local_module, IdentityLayer())
        self.local_module = local_module
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 定义前向传播过程
        x = self.context_module(x)  # 先通过context_module（包含LiteMSA和ResidualBlock）
        x = self.local_module(x)  # 再通过local_module（包含MBConv和ResidualBlock）
        return x  # 返回结果



#################################################################################
#                             Functional Blocks                                 #
#################################################################################


class ResidualBlock(nn.Module):
    def __init__(
            self,
            main: Optional[nn.Module],
            shortcut: Optional[nn.Module],
            post_act=None,
            pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res

class ResidualCrossBlock(nn.Module):
    def __init__(
            self,
            main: Optional[nn.Module],
            shortcut: Optional[nn.Module],
            post_act=None,
            pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualCrossBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor,edge: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x,edge)
        else:
            return self.main(self.pre_norm(x),self.pre_norm(edge))

    def forward(self, x: torch.Tensor,edge: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x,edge)
        else:
            res = self.forward_main(x,edge) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class DAGBlock(nn.Module):
    def __init__(
            self,
            inputs: Dict[str, nn.Module],
            merge_mode: str,
            post_input: Optional[nn.Module],
            middle: nn.Module,
            outputs: Dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge_mode = merge_mode
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        feat = merge_tensor(feat, self.merge_mode, dim=1)
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class OpSequential(nn.Module):
    def __init__(self, op_list: List[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


class EfficientViTBackbone(nn.Module):
    def __init__(self, width_list: List[int], depth_list: List[int], in_channels=1, dim=32, expand_ratio=4, norm="bn2d",
                 act_func="hswish") -> None:
        super().__init__()

        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)
        #self.channel = [i.size(1) for i in self.forward(torch.randn(8, 1, 512, 512))]

    @staticmethod
    def build_local_block(in_channels: int, out_channels: int, stride: int, expand_ratio: float, norm: str,
                          act_func: str, fewer_norm: bool = False) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(False, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(False, False, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        res = []
        x = self.input_stem(x)
        res.append(x)
        for stage_id, stage in enumerate(self.stages, 1):
            x = stage(x)
            res.append(x)
        return res


def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        k = k[9:]
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict


def efficientvit_b0(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']))
    return backbone


def efficientvit_b1(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        # depth_list=[1, 2, 3, 3, 4],
        depth_list=[1, 2, 2, 2, 1],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']))
    return backbone

def efficientvit_bs(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        # width_list=[16, 32, 64, 128],
        width_list=[32, 64, 128, 256],
        # depth_list=[1, 2, 3, 3, 4],
        depth_list=[1, 1, 1, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']),strict=False)
    return backbone

def efficientvit_bs16(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[16, 16, 16, 16],
        # depth_list=[1, 2, 3, 3, 4],
        depth_list=[1, 1, 1, 1],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']),strict=False)
    return backbone

def efficientvit_bs_32(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[32,64,128,256],
        # depth_list=[1, 2, 3, 3, 4],
        depth_list=[3, 3, 3, 3],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']))
    return backbone

def efficientvit_b2(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']))
    return backbone


def efficientvit_b3(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']))
    return backbone


if __name__ == '__main__':
    model = efficientvit_bs()
    inputs = torch.randn((1, 1, 512, 512))
    res = model(inputs)
    for i in res:
        print(i.size())