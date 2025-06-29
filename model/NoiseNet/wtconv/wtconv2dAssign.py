import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import wavelet





class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1', scale=1):
        super(WTConv2d, self).__init__()

        # 断言输入通道数和输出通道数相等
        assert in_channels == out_channels

        # 初始化输入通道数
        self.in_channels = in_channels
        # 初始化小波变换级数
        self.wt_levels = wt_levels
        # 初始化步长
        self.stride = stride
        # 初始化空洞卷积的空洞率
        self.dilation = 1
        self.scale = scale

        # 创建小波滤波器
        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        # 将小波滤波器转换为PyTorch中的Parameter，并设置requires_grad为False
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 初始化小波变换函数
        self.wt_function = wavelet.wavelet_transform_init(self.wt_filter)
        # 初始化逆小波变换函数
        self.iwt_function = wavelet.inverse_wavelet_transform_init(self.iwt_filter)

        # 初始化基础卷积层
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        # 初始化基础卷积层的缩放模块
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        # 初始化小波卷积层列表
        self.wavelet_convs = nn.ModuleList(
            # 循环创建多个小波卷积层
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        # 初始化小波卷积层的缩放模块列表
        self.wavelet_scale = nn.ModuleList(
            # 循环创建多个缩放模块，并设置初始缩放值为0.1
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        # 如果步长大于1，则进行步长处理
        if self.stride > 1:
            # 初始化步长滤波器
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            # 定义步长处理函数
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            # 步长处理函数为空
            self.do_stride = None

    def forward(self, x):
        # 存储每一级小波变换后的低频分量
        x_ll_in_levels = []
        # 存储每一级小波变换后的高频分量
        x_h_in_levels = []
        # 存储每一级小波变换前的输入形状
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            # 获取当前低频分量的形状
            curr_shape = curr_x_ll.shape
            # 将当前形状添加到列表中
            shapes_in_levels.append(curr_shape)
            # 如果当前低频分量的高度或宽度不是偶数，则进行填充
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
            # print("curr_x_ll", curr_x_ll.shape)
            # 对当前低频分量进行小波变换
            curr_x = self.wt_function(curr_x_ll)  # curr_x is 经过小波变换后的4通道的特征图
            # print("curr_x.shape: ", curr_x.shape)
            # 提取小波变换后的低频分量
            curr_x_ll = curr_x[:,:,0,:,:]

            # 获取当前小波变换后的形状
            shape_x = curr_x.shape
            # print("shape_x: ",shape_x)
            # 对高频分量进行形状变换
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            # print("curr_x_tag.shape: ",curr_x_tag.shape)
            # 对高频分量进行卷积和缩放
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            # print("curr_x_tag.shape: ", curr_x_tag.shape)
            # 将高频分量的形状恢复为原始形状
            curr_x_tag = curr_x_tag.reshape(shape_x)
            # print("curr_x_tag.shape: ", curr_x_tag.shape)

            # 将低频分量添加到列表中
            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            # 将高频分量添加到列表中
            x_h_in_levels.append(self.scale * curr_x_tag[:,:,1:4,:,:])
            # print(self.scale)

        next_x_ll = 0

        # 从最高级到最低级进行逆小波变换
        for i in range(self.wt_levels-1, -1, -1):
            # 弹出并获取当前低频分量
            curr_x_ll = x_ll_in_levels.pop()
            # 弹出并获取当前高频分量
            curr_x_h = x_h_in_levels.pop()
            # 弹出并获取当前形状
            curr_shape = shapes_in_levels.pop()

            # 将当前低频分量与下一级的低频分量相加
            curr_x_ll = curr_x_ll + next_x_ll

            # 将当前低频分量和高频分量拼接起来
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            # 对拼接后的分量进行逆小波变换
            next_x_ll = self.iwt_function(curr_x)

            # 根据原始形状裁剪逆小波变换后的低频分量
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        # 将最终的低频分量赋值给x_tag
        x_tag = next_x_ll
        # 断言确保x_ll_in_levels列表为空
        assert len(x_ll_in_levels) == 0

        # 对输入x进行基础卷积和缩放
        x = self.base_scale(self.base_conv(x))
        # 将基础卷积的结果与x_tag相加
        x = x + x_tag

        # 如果定义了步长处理函数，则对x进行步长处理
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        # 调用父类的初始化方法
        super(_ScaleModule, self).__init__()
        # 将传入的dims赋值给类的属性self.dims
        self.dims = dims
        # 创建一个参数weight，其形状与dims相同，初始值为init_scale
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        # 将self.bias设置为None
        self.bias = None
    
    def forward(self, x):
        # 将权重与输入进行乘法运算
        return torch.mul(self.weight, x)


def test_WTConv2d():
    # 设置输入张量的大小: (batch_size, in_channels, height, width)
    input_tensor = torch.randn(1, 16, 64, 64)  # 例如 batch_size=1, in_channels=16, 高度和宽度为64

    # 创建 WTConv2d 实例
    wt_conv = WTConv2d(in_channels=16, out_channels=16, kernel_size=3, wt_levels=1, stride=1, wt_type='db1')

    # 将输入张量通过 WTConv2d 层
    output_tensor = wt_conv(input_tensor)

    # 打印输出张量的形状
    print("Output Tensor Shape:", output_tensor.shape)

# 运行测试
# test_WTConv2d()