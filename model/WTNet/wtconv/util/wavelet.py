import pywt
import pywt.data
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    # 创建一个小波对象
    w = pywt.Wavelet(wave)
    # 将分解滤波器的高通部分转换为PyTorch张量，并进行反转
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    # 将分解滤波器的低通部分转换为PyTorch张量，并进行反转
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    # 创建分解滤波器矩阵
    dec_filters = torch.stack([
        # 低通滤波器与低通滤波器的乘积
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        # 低通滤波器与高通滤波器的乘积
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        # 高通滤波器与低通滤波器的乘积
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        # 高通滤波器与高通滤波器的乘积
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)

    # 将分解滤波器矩阵在第一个维度上增加一个维度，并重复in_size次
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    # 将重构滤波器的高通部分转换为PyTorch张量，并进行反转和翻转
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    # 将重构滤波器的低通部分转换为PyTorch张量，并进行反转和翻转
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    # 创建重构滤波器矩阵
    rec_filters = torch.stack([
        # 低通滤波器与低通滤波器的乘积
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        # 低通滤波器与高通滤波器的乘积
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        # 高通滤波器与低通滤波器的乘积
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        # 高通滤波器与高通滤波器的乘积
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)

    # 将重构滤波器矩阵在第一个维度上增加一个维度，并重复out_size次
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    # 返回分解滤波器和重构滤波器
    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    # 获取输入x的形状
    b, c, h, w = x.shape

    # 计算需要填充的像素值
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)

    # x: 输入张量。
    # filters: 小波变换的滤波器（即卷积核）。
    # stride = 2: 步长为2，表示每次卷积窗口移动两个像素，这样输出的高度和宽度会是原来的一半。
    # groups = c: 分组卷积，将输入的每个通道独立地与相应的滤波器进行卷积。
    # padding = pad: 在输入张量的边界添加填充像素，以保持卷积后的尺寸。
    # 对输入x进行二维卷积操作，并设置步长为2，分组为c，填充为pad
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)

    # 卷积操作完成后，输出的张量x形状变为[b, c, h // 2, w // 2]。
    # 由于每个通道经过小波变换会产生4个子带（LL, LH, HL, HH），需要将输出重新塑形为[b, c, 4, h // 2, w // 2]。
    # 其中4表示4个子带，分别对应于低频和高频的组合。
    # 将卷积后的结果重新塑形为 (b, c, 4, h // 2, w // 2)
    x = x.reshape(b, c, 4, h // 2, w // 2)

    return x


def inverse_wavelet_transform(x, filters):
    # 这行代码获取输入张量x的形状信息。
    # b表示批量大小，c表示通道数，_表示子带数（固定为4），h_half和w_half分别表示每个子带的高度和宽度的一半。
    # 获取输入x的形状
    b, c, _, h_half, w_half = x.shape

    # 计算需要填充的像素值
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)

    # 将输入张量x重新塑形为[b, c * 4, h_half, w_half]，这样每个通道的子带会被展平成单个通道。
    # 将x重新塑形为 (b, c*4, h_half, w_half)
    x = x.reshape(b, c * 4, h_half, w_half)

    # 对x进行二维转置卷积操作，并设置步长为2，分组为c，填充为pad
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)

    return x


def wavelet_transform_init(filters):
    class WaveletTransform(Function):

        # 前向传播
        @staticmethod
        def forward(ctx, input):
            # 在不计算梯度的模式下进行小波变换
            with torch.no_grad():
                x = wavelet_transform(input, filters)
            return x

        # 反向传播
        @staticmethod
        def backward(ctx, grad_output):
            # 对梯度进行逆小波变换
            grad = inverse_wavelet_transform(grad_output, filters)
            # 返回梯度，第二个返回值为None
            return grad, None

    return WaveletTransform().apply


def inverse_wavelet_transform_init(filters):
    class InverseWaveletTransform(Function):

        # 前向传播函数
        @staticmethod
        def forward(ctx, input):
            # 在没有梯度的情况下执行inverse_wavelet_transform函数
            with torch.no_grad():
                x = inverse_wavelet_transform(input, filters)
            return x

        # 反向传播函数
        @staticmethod
        def backward(ctx, grad_output):
            # 对grad_output执行wavelet_transform函数，得到梯度grad
            grad = wavelet_transform(grad_output, filters)
            # 返回梯度grad和None（因为该层没有需要更新的参数）
            return grad, None

    # 返回InverseWaveletTransform类的apply方法，用于在PyTorch中调用
    return InverseWaveletTransform().apply
