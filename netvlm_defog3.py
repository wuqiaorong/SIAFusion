import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import MaxPool2d
import torch.nn.functional as F
import numpy as np
import test_othe_model.fusion_strategy as fusion_strategy
from restormer import TransformerBlock as Restormer, cross_TransformerBlock, cross_TransformerBlock_single
from utils import stretchImage,getPara,zmIce,zmIceFast,zmIceColor1
import cv2
import math
# 根据半径计算权重参数矩阵
g_para = {}
class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) +
                         self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / \
                (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / \
                (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate



class DownsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, is_last=False):
        super(DownsampleConvLayer, self).__init__()
        self.is_last = is_last
        self.sigmod = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 计算填充
        padding = (kernel_size - 1) // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding)

    def forward(self, x):
        out = self.conv2d(x)
        # print("out shape:", out.shape)
        if self.is_last is False:
            out = self.bn(out)
            out = self.lrelu(out)
        else:
            out = self.sigmod(out)
        return out

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=2, out_chans=3, embed_dim=128, kernel_size=3):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)  # 利用输入边界的反射来填充输入张量

        # 使用最近邻插值将特征图尺寸扩大 patch_size 倍
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False)
        # self.upsample = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 卷积层，输入通道数为 embed_dim，输出通道数为 out_chans
        self.conv = nn.Conv2d(embed_dim, out_chans, kernel_size=kernel_size,stride=1)
        self.conv_re=ConvLayer(out_chans, out_chans, kernel_size=1,stride=1)

    def forward(self, x):
        # 先进行最近邻插值
        x = self.upsample(x)
        # 再进行卷积操作
        x = self.reflection_pad(x)
        x = self.conv(x)
        x=self.lrelu(x)
        x=self.conv_re(x)+x
        return x

# Convolution operation卷积运算
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)#利用输入边界的反射来填充输入张量
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last
        # self.tahn1=nn.Tanh()
        self.sigmod=nn.Sigmoid()
        self.bn=nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out=self.bn(out)
            out = self.lrelu(out)
            # out=F.relu(out)
            # out = self.dropout(out)
        else:
            out=self.sigmod(out)
        return out


# Convolution operation卷积运算
class ConvLayer_fusion(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer_fusion, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)#利用输入边界的反射来填充输入张量
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last
        # self.tahn1=nn.Tanh()
        self.bn=nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out=self.bn(out)
            out = self.lrelu(out)
            # out = self.dropout(out)
        else:
            out=self.tahn1(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Cross_WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2 , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C), which maps query
            y: input features with shape of (num_windows*B, N, C), which maps key and value
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias 定义相对位置偏差的参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))#@表示矩阵乘法

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class Cross_SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1_A = norm_layer(dim)
        self.norm1_B = norm_layer(dim)
        self.attn_A = Cross_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn_B = Cross_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path_A = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_B = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_A = norm_layer(dim)
        self.norm2_B = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_A = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_B = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA计算掩膜，将不能注意力计算的部分填充-100
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, y):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut_A = x
        shortcut_B = y
        x = self.norm1_A(x)
        y = self.norm1_B(y)
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        y_windows = window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, C
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows_A = self.attn_A(x_windows, y_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        attn_windows_B = self.attn_B(y_windows, x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows_A = attn_windows_A.view(-1, self.window_size, self.window_size, C)
        attn_windows_B = attn_windows_B.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_A, self.window_size, H, W)  # B H' W' C
        shifted_y = window_reverse(attn_windows_B, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            y = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            y = shifted_y
        x = x.view(B, H * W, C)
        y = y.view(B, H * W, C)

        # FFN
        x = shortcut_A + self.drop_path_A(x)
        x = x + self.drop_path_A(self.mlp_A(self.norm2_A(x)))

        y = shortcut_B + self.drop_path_B(y)
        y = y + self.drop_path_B(self.mlp_B(self.norm2_B(y)))
        return x, y

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


#STL
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear',token_mlp='ffn',se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA计算掩膜，将不能注意力计算的部分填充-100
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift,将不相邻的patch合并
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
    def flops(self):
        flops = 0
        H, W = self.HEIGTH, self.WIDTH
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops



class Cross_BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            Cross_SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # # patch merging layer
        # if downsample is not None:
        #     self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        # else:
        #     self.downsample = None
    # 以x为主, y只是辅助x的特征提取
    def forward(self, x, y):
        for blk in self.blocks:
            if self.use_checkpoint:
                x, y = checkpoint.checkpoint(blk, x, y)
            else:
                x, y = blk(x, y)
        # if self.downsample is not None:
        #     x = self.downsample(x)
        #     y = self.downsample(y)
        # print("Cross_BasicLayer:", type(x), type(y))
        return x, y

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops



#RSTblock
class StageModule(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class restormer_block(nn.Module):
    def __init__(self, restormerdim,
                 restormerhead,
                 ffn_expansion_factor,
                 depth,
                 bias=False,
                 LayerNorm_type='WithBias',
                drop=0., attn_drop=0.,
                 drop_path=0.,use_checkpoint=False):

        super().__init__()
        self.restormerdim = restormerdim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            Restormer(restormerdim, restormerhead, ffn_expansion_factor, bias, LayerNorm_type,
                                 proj_drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 )
            for i in range(self.depth)])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            return x

    # def extra_repr(self) -> str:
    #     return f"dim={self.restormerdim},  depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
class cross_restormer_block(nn.Module):
    def __init__(self, restormerdim,
                 restormerhead,
                 ffn_expansion_factor,
                 depth,
                 bias=False,
                 LayerNorm_type='WithBias',
                drop=0., attn_drop=0.,
                 drop_path=0.,use_checkpoint=False):

        super().__init__()
        self.restormerdim = restormerdim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            cross_TransformerBlock(restormerdim, restormerhead, ffn_expansion_factor, bias, LayerNorm_type,
                                 proj_drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 )
            for i in range(self.depth)])

    def forward(self, x,y):
        for blk in self.blocks:
            if self.use_checkpoint:
                x,y = checkpoint.checkpoint(blk, x,y)
            else:
                x,y = blk(x,y)
            return x,y

    # def extra_repr(self) -> str:
    #     return f"dim={self.restormerdim},  depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
class cross_restormer_block_single(nn.Module):
    def __init__(self, restormerdim,
                 restormerhead,
                 ffn_expansion_factor,
                 depth,
                 bias=False,
                 LayerNorm_type='WithBias',
                drop=0., attn_drop=0.,
                 drop_path=0.,use_checkpoint=False):

        super().__init__()
        self.restormerdim = restormerdim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            cross_TransformerBlock_single(restormerdim, restormerhead, ffn_expansion_factor, bias, LayerNorm_type,
                                 proj_drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 )
            for i in range(self.depth)])

    def forward(self, x,y):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x,y)
            else:
                x = blk(x,y)
            return x

    # def extra_repr(self) -> str:
    #     return f"dim={self.restormerdim},  depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

#嵌入层，将初始特征l重塑为序列向量
class PatchEmbed(nn.Module):
    def __init__(self, img_size=420, patch_size=1, in_chans=1, embed_dim=512, norm_layer=None,init_dim1=[32,64,128,256,512]):
        super().__init__()
        img_size = to_2tuple(img_size)#将输入转为元组,新元组长度为2
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        # self.init_dim=init_dim
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # # #在嵌入层使用卷积增加效果
        self.swincov=nn.Sequential(ConvLayer(init_dim1[0],init_dim1[1],kernel_size=3,stride=1),
                                   ConvLayer(init_dim1[1], init_dim1[2], kernel_size=3, stride=1),
                                   ConvLayer(init_dim1[2], init_dim1[3], kernel_size=3, stride=1),
                                   ConvLayer(init_dim1[3], init_dim1[4], kernel_size=3, stride=1),
                                   # ConvLayer(init_dim[2], init_dim[3], kernel_size=3, stride=1),
                                   )
        self.proj = nn.Conv2d(in_chans, init_dim1[0], kernel_size=patch_size, stride=patch_size)#通道承接像素信息
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)#得到位置编码
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        x=self.swincov(x)
        # print(x.shape)
        x = x.flatten(2).transpose(1, 2)  # 操作之后的维度：[B,Ph*Pw,C]，降维＋转置
        if self.norm is not None:
            x = self.norm(x)#归一化，96维
        # print(x.shape)
        return x
    #计算计算复杂度
    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0),-1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
class CrossAttention_vlm(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention_vlm, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = attn_output.transpose(0, 1)

        return attn_output
class VLMattention(nn.Module):
    def __init__(self,
            embed_dims,num_heads,drop_path=0.,
                 attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.convA2 = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.preluA2 = nn.PReLU()
        self.convA3 = nn.Conv2d(2 * embed_dims, embed_dims, kernel_size=1)
        self.preluA3 = nn.PReLU()
        # self.imagef2textfA1 = imagefeature2textfeature(embed_dims[1], embed_dims[1], embed_dims[2])
        self.cross_attentionA1 = CrossAttention_vlm(embed_dim=embed_dims, num_heads=num_heads)
        self.image2text_dim = embed_dims
        self.bn = nn.BatchNorm2d(embed_dims)
        self.drop_path_A = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm=nn.LayerNorm(embed_dims)
    def untransformer(self, x):

        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)

        return x
    def forward(self, x_vi_1,text_feature):
        b,_,H,W=x_vi_1.shape
        imageAtotext = self.untransformer(x_vi_1)
        imageA_sideout = x_vi_1
        # print(imageA_sideout)
        # print(text_feature.shape)
        ca_A = self.cross_attentionA1(text_feature, imageAtotext, imageAtotext)
        # print(ca_A.shape)
        # print(imageAtotext.shape)
        # ca_A = ca_A + text_feature
        ca_A = torch.nn.functional.adaptive_avg_pool1d(ca_A.permute(0, 2, 1), 1).permute(0, 2, 1)
        ca_A = F.normalize(ca_A, p=1, dim=2)
        ca_A = (imageAtotext * ca_A).view(b, self.image2text_dim, H, W)
        # print(ca_A.shape)
        # imageA_sideout = F.interpolate(imageA_sideout, [H, W], mode='nearest')
        # ca_A = F.interpolate(ca_A, [H, W], mode='nearest')
        # print(ca_A.shape)
        ca_A = self.preluA3(
            self.convA3(torch.cat(
                (x_vi_1, self.preluA2(self.convA2(ca_A)) + imageA_sideout), 1)))
        ca_A=self.bn(self.drop_path_A(ca_A)+x_vi_1)
        return ca_A
class VLMattention_block(nn.Module):
    def __init__(self, embed_dims,num_heads,depth,drop=0., attn_drop=0.,
                 drop_path=0.,use_checkpoint=False):

        super().__init__()
        self.embed_dims = embed_dims
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            VLMattention(embed_dims=embed_dims,num_heads=num_heads,proj_drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 )
            for i in range(self.depth)])

    def forward(self, x,y):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x,y)
            else:
                x = blk(x,y)
            return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        # x_out=x+x_out
        return x_out


class CLIP_CAM_Generator(nn.Module):
    def __init__(self, text_code=77, img_dim=512):
        super().__init__()
        # 注意力权重生成
        self.attn_pool = nn.AdaptiveAvgPool2d(1)
        self.text_weight = WeightedChannelReducer(num_words=text_code, use_softmax=True)

        # self.fc = nn.Linear(img_dim, 1)
        # self.sigmod=nn.Sigmoid()
        # self.cbam = CBAM(text_code)

    def forward(self, text_feats, img_feats):

        sim_map = torch.einsum('bchw,blc->blhw', img_feats, text_feats)  # [1,77,W,H]
        # print(sim_map)
        #简单注意力进行通道排序
        # sim_map1 = sim_map
        # sim_map = self.cbam(sim_map)
        # print(text_feats)
        sim_map=self.text_weight(sim_map)
        # sim_map = sim_map.mean(dim=1)
        # 生成类激活图
        # cam = self.fc(self.attn_pool(img_feats).flatten(1)) * sim_map
        # return self.sigmod(cam),sim_map.softmax(dim=-1),text_feats  # 归一化到[0,1]
        # return sim_map.softmax(dim=-1), sim_map.softmax(dim=-1), text_feats  # 归一化到[0,1]
        # sim_map_norm=sim_map
        sim_map_norm = (sim_map - torch.min(sim_map)) / (
                torch.max(sim_map) - torch.min(sim_map)+ 1e-8 )
        # print(sim_map)
        return sim_map_norm, sim_map_norm, text_feats  # 归一化到[0,1]
class LOSS_CAM_fusion(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, text_feats, img_feats):
        # print(text_feats.shape, img.shape)
        """
        输入:
            text_feats: [1,77,512] CLIP文本特征
            img: [1,3,224,224] 输入图像
        输出:
            cam: [1,224,224] 单通道激活图
        """
        # 文本特征聚合（参考网页8的文本编码器）
        text_emb = text_feats
        # 跨模态注意力计算（参考网页9的对比学习思想）
        sim_map = torch.einsum('bchw,blc->blhw', img_feats, text_emb)  # [1,224,224]
        # print(sim_map.shape)
        sim_map = sim_map.mean(dim=1)
        # print(sim_map.softmax(dim=-1).shape)
        return sim_map.softmax(dim=-1)  # 归一化到[0,1]

class LearnablePositionEncoder(nn.Module):
    def __init__(self, in_dim=1, out_dim=420, target_size=105):
        super().__init__()
        # 升维模块
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 256, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(256, out_dim, kernel_size=3, padding=1)
        )
        # 自适应下采样
        self.downsample = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            # nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
            # nn.GELU(),
            # nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            # nn.GELU(),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, cam):
        # print(cam.shape)
        """
        输入:
            cam: [1,420,420] 类激活图
        输出:
            pos_enc: [1,105,105,128] 位置编码
        """
        # 升维到128通道（参考网页5的特征增强）
        x = self.upscale(cam.unsqueeze(1))  # [1,128,420,420]
        # print(x.shape)

        # 下采样到目标尺寸（参考网页9的稳健编码设计）
        x = self.downsample(x)  # [1,128,105,105]
        # print(x.shape)
        return x  # [1,128,105,105]

#文本图像交互后特征融合而不简单平均
class WeightedChannelReducer(nn.Module):
    def __init__(self, num_words, use_softmax=True):
        """
        Args:
            num_words (int): 单词数量，即特征图的通道数 L
            use_softmax (bool): 是否对权重应用Softmax归一化（默认启用）
        """
        super().__init__()
        self.num_words = num_words
        self.use_softmax = use_softmax

        # 初始化可学习权重（初始值为均等权重）
        self.weight = nn.Parameter(torch.ones(num_words) / num_words)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        """
        Args:
            x: 输入特征图，形状为 [W, H, L] 或 [batch, W, H, L]
        Returns:
            输出特征图，形状为 [W, H] 或 [batch, W, H]
        """
        # 确保输入形状有效
        assert x.size(-1) == self.num_words, f"Last dim must be {self.num_words}, got {x.shape}"

        # 处理批量数据（若存在batch维度）
        has_batch = len(x.shape) == 4
        if has_batch:
            batch_size = x.shape[0]
            # 将特征图展平为 [batch, W*H, L]
            x_flat = x.reshape(batch_size, -1, self.num_words)
        else:
            # 无batch：展平为 [W*H, L]
            x_flat = x.reshape(-1, self.num_words)

        # 权重归一化（可选）
        weights = self.weight
        if self.use_softmax:
            weights = torch.softmax(weights, dim=-1)  # 权重非负且和为1

        # 计算加权平均：x_flat @ weights -> [batch, W*H] 或 [W*H]
        output_flat = torch.matmul(x_flat, weights)

        # 恢复空间维度
        spatial_dims = x.shape[:-1]  # 原始空间维度 (W, H) 或 (batch, W, H)
        output = output_flat.view(spatial_dims)
        return output


#最终模型
class SwinFuse(nn.Module):
    def __init__(self, img_size=420, patch_size=8, in_chans=1, out_chans=1,
                 embed_dim=512, depths=[6, 6, 6, 6], num_heads=[8,8,8,8],bias=False,LayerNorm_type='WithBias',
                mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                 use_checkpoint=False, init_dim=[128,64,32,16,1],**kwargs):
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};drop_path_rate:{};out_chans:{}".format(
                depths,
                drop_path_rate, out_chans))

        self.patch_size = patch_size
        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim)
        self.num_features_up = int(embed_dim)
        self.mlp_ratio = mlp_ratio


        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        #减少过拟合
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth随机深度，在[0.0.2]之间等差取了18个点
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        #  encoder

        self.EN1_0 = restormer_block(restormerdim=int(embed_dim),
                                 restormerhead=num_heads[0],
                                 depth=depths[0],
                                ffn_expansion_factor=self.mlp_ratio,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                bias=bias,LayerNorm_type=LayerNorm_type,
                                 drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                                 use_checkpoint=use_checkpoint)
        self.EN2_0 = restormer_block(restormerdim=int(embed_dim),
                                     restormerhead=num_heads[1],
                                     depth=depths[1],
                                     ffn_expansion_factor=self.mlp_ratio,
                                     drop=drop_rate, attn_drop=attn_drop_rate,
                                     bias=bias, LayerNorm_type=LayerNorm_type,
                                     drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                                     use_checkpoint=use_checkpoint)
        self.EN3_0 = restormer_block(restormerdim=int(embed_dim),
                                     restormerhead=num_heads[2],
                                     depth=depths[2],
                                     ffn_expansion_factor=self.mlp_ratio,
                                     drop=drop_rate, attn_drop=attn_drop_rate,
                                     bias=bias, LayerNorm_type=LayerNorm_type,
                                     drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
                                     use_checkpoint=use_checkpoint)
        #  decoder
        self.norm = nn.BatchNorm2d(self.num_features)
        self.apply(self._init_weights)
        self.swincov_re = nn.Sequential(
            PatchUnEmbed(patch_size=2, out_chans=init_dim[0], embed_dim=embed_dim, kernel_size=3),
            PatchUnEmbed(patch_size=2, out_chans=init_dim[1], embed_dim=init_dim[0], kernel_size=3),
            PatchUnEmbed(patch_size=2, out_chans=init_dim[2], embed_dim=init_dim[1], kernel_size=3),
            # ConvLayer(init_dim[0], init_dim[1], kernel_size=3, stride=1),
            ConvLayer(init_dim[2], init_dim[3], kernel_size=3, stride=1),
            ConvLayer(init_dim[3], init_dim[4], kernel_size=3, stride=1, is_last=True),
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #embed
    def self_embed(self,x,h,w):

        x=self.patch_embed(x)
        B, L, C = x.shape

        # print(x.shape)
        # print(self.absolute_pos_embed.shape)
        if self.ape:
            if self.absolute_pos_embed.shape[1] != L:
                # 使用插值方法调整位置编码的形状
                absolute_pos_embed = F.interpolate(self.absolute_pos_embed.unsqueeze(0), size=(L, C), mode='bilinear',
                                                   align_corners=False).squeeze(0)
                x = x + absolute_pos_embed
            else:
                absolute_pos_embed=self.absolute_pos_embed
                x = x + absolute_pos_embed
        x = self.pos_drop(x)
        x = self.transformer(x, h // self.patch_size, w // self.patch_size)
        absolute_pos_embed=self.transformer(absolute_pos_embed, h // self.patch_size, w // self.patch_size)
        return x,absolute_pos_embed
    # Encoder
    def encoder(self, x):
        x1 = self.EN1_0(x)
        x1 = x1+x
        x2 = self.EN2_0(x1)
        x2 = x2+x1
        x2 = self.norm(x2)
        return x2
    def encoder1(self,x):
        x1=self.EN1_0(x)
        x1=x1+x
        x1=self.norm(x1)
        return x1
    def encoder2(self,x):
        x1=self.EN2_0(x)
        x1=x1+x
        x1=self.norm(x1)
        return x1

    # transform，reshape
    def transformer(self, x,h,w):

        B, L, C = x.shape
        H=h
        W=w
        # H = W = int(L ** 0.5)
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        return x

    # untransformer
    def untransformer(self, x):

        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)

        return x

    # Fusion
    def fusion1(self, en1, en2, p_type):
        # attention weight
        fusion_function = fusion_strategy.attention_fusion_weight

        f = fusion_function(en1, en2, p_type)
        return f

    # Dencoder
    def up_x4(self, x2):
        # x1 = self.untransformer(x1)
        # x2 = self.EN2_0(x1)
        # x2 = x2 + x1
        # x2=self.untransformer(x2)
        # x3 = self.EN3_0(x2)
        # x3 = x3 + x2
        x3 = self.EN3_0(x2)
        # x3 = self.untransformer(x3)
        # x4 = self.EN4_0(x3)
        # x4 = x3 + x4
        x3 = self.norm(x3)
        # x3 = self.transformer(x3)
        x3=self.swincov_re(x3)
        # x = self.conv(x)
        # x = self.tanh(x)
        return x3

    def finaldecoder(self, x,h,w):

        x_e = self.encoder(x,h,w)
        x_d = self.up_x4(x_e)

        return x_d

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (
                    2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
class TextCorrespond(nn.Module):
    def __init__(self, dim, text_channel, amplify=2):
        super(TextCorrespond, self).__init__()

        #d = max(int(dim/reduction), 4)
        d = int(dim*amplify)
        self.mlp_text_features = nn.Sequential(
            nn.Conv1d(text_channel, d, 1),
            nn.LeakyReLU(),
            nn.Conv1d(d, dim, 1, bias=False),
            nn.LeakyReLU(),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self,  text_features):

        combined_features = self.mlp_text_features(text_features.permute(0, 2, 1).float())
        combined_features=self.ln(combined_features.permute(0, 2, 1))
        return combined_features

class FusionBlock_res(torch.nn.Module):
    def __init__(self, embed_dim=512, out_chans=1,depths=[2,1,2,1,1], num_heads=[8,8,8,8],amplify=2,text_channel=512,
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,img_size=224,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer_1=nn.BatchNorm2d, norm_layer_2=nn.LayerNorm,ape=True, patch_norm=True,bias=False,LayerNorm_type='WithBias',
                 use_checkpoint=False, **kwargs):
        super(FusionBlock_res, self).__init__()
        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim)
        self.num_features_up = int(embed_dim)
        self.mlp_ratio = mlp_ratio
        self.patch_size = 8
        self.target_size = img_size//self.patch_size
        self.TextCorrespond=TextCorrespond(embed_dim,text_channel,amplify)
        if self.ape:
            self.cam = CLIP_CAM_Generator(text_code=77,img_dim=512)
            # self.learn_pos = LearnablePositionEncoder(in_dim=1, out_dim=embed_dim, target_size=self.target_size)
        # 减少过拟合
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.loss_cam1 = LOSS_CAM_fusion()
        # self.text_weight=WeightedChannelReducer(num_words=77, use_softmax=True)

        # stochastic depth随机深度，在[0.0.2]之间等差取了18个点
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.cbam_vi = nn.Sequential(#ConvLayer(1, embed_dim//2, kernel_size=3, stride=1),
                                     #ConvLayer(embed_dim//2, embed_dim, kernel_size=3, stride=1),
                                     CBAM(embed_dim),
                                     CBAM(embed_dim),
                                     ConvLayer(embed_dim, embed_dim, kernel_size=1, stride=1),

                                     )
        self.cbam_ir = nn.Sequential(#ConvLayer(1, embed_dim//2, kernel_size=3, stride=1),
                                     #ConvLayer(embed_dim//2, embed_dim, kernel_size=3, stride=1),
                                     CBAM(embed_dim),
                                     CBAM(embed_dim),
                                     ConvLayer(embed_dim, embed_dim, kernel_size=1, stride=1),
                                     )
        self.cov_map = nn.Sequential(ConvLayer(2 * embed_dim, embed_dim, kernel_size=3, stride=1),
                                     # ConvLayer(embed_dim, embed_dim // 2, kernel_size=3, stride=1),
                                     ConvLayer(embed_dim, embed_dim, kernel_size=1, stride=1),
                                     )
        self.sigmod = nn.Sigmoid()
        self.conv_fusion = nn.Sequential(ConvLayer_fusion(2*self.embed_dim, self.embed_dim, 3, 1),
                                         ConvLayer_fusion(self.embed_dim, self.embed_dim, 1, 1) )
        # self.conv_fusion_word = nn.Sequential(ConvLayer_fusion(self.embed_dim, self.embed_dim, 1, 1))
        # self.conv_cam=nn.Sequential(ConvLayer_fusion(self.embed_dim, 2 * self.embed_dim, 3, 1),
        #                                  ConvLayer_fusion(2 * self.embed_dim, self.embed_dim, 3, 1),
        #                             ConvLayer_fusion(self.embed_dim, self.embed_dim, 1, 1))
        # self.fusion_cam = nn.Sequential(ConvLayer_fusion(self.embed_dim, 2 * self.embed_dim, 3, 1),
        #                               ConvLayer_fusion(2 * self.embed_dim, self.embed_dim, 3, 1),
        #                               ConvLayer_fusion(self.embed_dim, self.embed_dim, 1, 1))
        self.GCT=GCT(self.embed_dim)
        # self.GCT_ir =GCT(self.embed_dim)
        self.cam_text=nn.Sequential(ConvLayer_fusion(2 *self.embed_dim,  self.embed_dim, 3, 1),
                                         )
        self.final_conv=ConvLayer_fusion(self.embed_dim, self.embed_dim, 1, 1)
        self.cross_module_A1=cross_restormer_block_single(restormerdim=int(embed_dim),
                                                            restormerhead=num_heads[1],
                                                            depth=depths[0],
                                                            ffn_expansion_factor=self.mlp_ratio,
                                                            drop=drop_rate, attn_drop=attn_drop_rate,
                                                            bias=bias, LayerNorm_type=LayerNorm_type,
                                                            drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                                                            use_checkpoint=use_checkpoint)
        self.cross_module_B1 =cross_restormer_block_single(restormerdim=int(embed_dim),
                                                            restormerhead=num_heads[1],
                                                            depth=depths[0],
                                                            ffn_expansion_factor=self.mlp_ratio,
                                                            drop=drop_rate, attn_drop=attn_drop_rate,
                                                            bias=bias, LayerNorm_type=LayerNorm_type,
                                                            drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                                                            use_checkpoint=use_checkpoint)
        self.cross_module_A2 = cross_restormer_block_single(restormerdim=int(embed_dim),
                                                            restormerhead=num_heads[1],
                                                            depth=depths[1],
                                                            ffn_expansion_factor=self.mlp_ratio,
                                                            drop=drop_rate, attn_drop=attn_drop_rate,
                                                            bias=bias, LayerNorm_type=LayerNorm_type,
                                                            drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                                                            use_checkpoint=use_checkpoint)
        self.cross_module_B2 = cross_restormer_block_single(restormerdim=int(embed_dim),
                                                            restormerhead=num_heads[1],
                                                            depth=depths[1],
                                                            ffn_expansion_factor=self.mlp_ratio,
                                                            drop=drop_rate, attn_drop=attn_drop_rate,
                                                            bias=bias, LayerNorm_type=LayerNorm_type,
                                                            drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                                                            use_checkpoint=use_checkpoint)
        self.cross_module_1 = cross_restormer_block(restormerdim=int(embed_dim),
                                                    restormerhead=num_heads[2],
                                                    depth=depths[2],
                                                    ffn_expansion_factor=self.mlp_ratio,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    bias=bias, LayerNorm_type=LayerNorm_type,
                                                    drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
                                                    use_checkpoint=use_checkpoint)

        self.cross_module_2_A = cross_restormer_block_single(restormerdim=int(embed_dim),
                                                    restormerhead=num_heads[2],
                                                    depth=depths[3],
                                                    ffn_expansion_factor=self.mlp_ratio,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    bias=bias, LayerNorm_type=LayerNorm_type,
                                                    drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
                                                    use_checkpoint=use_checkpoint)
        self.cross_module_2_B = cross_restormer_block_single(restormerdim=int(embed_dim),
                                                             restormerhead=num_heads[2],
                                                             depth=depths[3],
                                                             ffn_expansion_factor=self.mlp_ratio,
                                                             drop=drop_rate, attn_drop=attn_drop_rate,
                                                             bias=bias, LayerNorm_type=LayerNorm_type,
                                                             drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
                                                             use_checkpoint=use_checkpoint)
        # self.cross_module_final = cross_restormer_block_single(restormerdim=int(embed_dim),
        #                                                      restormerhead=num_heads[2],
        #                                                      depth=depths[3],
        #                                                      ffn_expansion_factor=self.mlp_ratio,
        #                                                      drop=drop_rate, attn_drop=attn_drop_rate,
        #                                                      bias=bias, LayerNorm_type=LayerNorm_type,
        #                                                      drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
        #                                                      use_checkpoint=use_checkpoint)
        # self.cross_module_pre_2 = cross_restormer_block_single(restormerdim=int(embed_dim),
        #                                                      restormerhead=num_heads[2],
        #                                                      depth=depths[3],
        #                                                      ffn_expansion_factor=self.mlp_ratio,
        #                                                      drop=drop_rate, attn_drop=attn_drop_rate,
        #                                                      bias=bias, LayerNorm_type=LayerNorm_type,
        #                                                      drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
        #                                                      use_checkpoint=use_checkpoint)
        # self.mlp_vis=nn.Sequential(nn.Conv2d(embed_dim, embed_dim*2, kernel_size=1,bias=False),
        #                            nn.PReLU(),
        #                            nn.Conv2d(embed_dim* 2, embed_dim, kernel_size=1, bias=False),
        #                            nn.PReLU(),
        #                            )
        # self.mlp_ir = nn.Sequential(nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=1, bias=False),
        #                              nn.PReLU(),
        #                              nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, bias=False),
        #                              nn.PReLU(),
        #                              )
        self.apply(self._init_weights)
        self.norm_1_1 = norm_layer_1(self.num_features)
        self.norm_1_2 = norm_layer_1(self.num_features)
        self.norm_1_3 = norm_layer_1(self.num_features)
        self.norm_1_4 = norm_layer_1(self.num_features)
        self.norm_1_5 = norm_layer_1(self.num_features)
        self.norm_1_6 = norm_layer_1(self.num_features)
        self.norm_1_7 = norm_layer_1(self.num_features)
        self.norm_2 = norm_layer_2(self.num_features)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    # transform，reshape
    def transformer(self, x, h, w):

        B, L, C = x.shape
        H = h
        W = w
        # H = W = int(L ** 0.5)
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        return x

        # untransformer
    def untransformer(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm_2(x)

        return x
    def forward(self, x_ir, x_vi, pos_vi,pos_ir):
        #直接将文本聚合，减少特征传递造成的匹配1信息减弱
        f_cross_Q=pos_vi*x_vi+pos_ir*x_ir
        # f_cross_Q=self.conv_fusion_word(f_cross_Q)
        f_cross_Q=self.norm_1_1(f_cross_Q)
        #图像特征互注意力计算
        f_cross_1_vi=self.cross_module_A2(x_vi,f_cross_Q)
        f_cross_2_ir=self.cross_module_B2(x_ir,f_cross_Q)
        f_cross_1_vi=self.norm_1_3(f_cross_1_vi+x_vi)
        f_cross_2_ir=self.norm_1_4(f_cross_2_ir+x_ir)
        #图像特征聚合
        f_cross_vi,f_cross_ir=self.cross_module_1(f_cross_1_vi,f_cross_2_ir)
        f_cross_Q2=self.norm_1_2(f_cross_vi+f_cross_ir)
        #以聚合注意力为查询向量再算一下注意力
        f_cross_vi_2=self.cross_module_2_A(f_cross_vi,f_cross_Q2)
        f_cross_ir_2=self.cross_module_2_B(f_cross_ir,f_cross_Q2)
        # #注意力合并
        out = torch.cat([f_cross_vi_2, f_cross_ir_2], 1)
        out = self.conv_fusion (out)+self.GCT(f_cross_Q)
        #合并后再作一次自注意力和1x1卷积
        out=self.norm_1_5(out)
        return out
    def encoder_text(self,x,y):
        # print(x.shape)
        # print(y.shape)
        cam,sim,cos = self.cam(y, x)
        # cam_1=self.text_weight(cam)+cam
        # cam_2=
        # pos = self.learn_pos(cam)
        return cam,sim,cos,cam
        # return cam, sim, cos
    def loss_cam(self,x,y):
        loss_cam=self.loss_cam1(x,y)
        return loss_cam
    # def encoder_vi(self, x_vi):
    #     x_vi = self.mlp_vis(x_vi)
    #     return x_vi
    #
    # def encoder_ir(self, x_ir):
    #     x_ir = self.mlp_ir(x_ir)
    #     return x_ir
    def img_dazing(self, x_vi, x_ir):
        # x_vi=self.transformer(x_vi)
        # x_ir=self.transformer(x_ir)
        x_vi_1 = self.cbam_vi(x_vi)
        x_ir_1 = self.cbam_ir(x_ir)
        x_vi_2 = torch.cat([x_ir_1, x_vi_1], 1)
        x_vi_map = self.cov_map(x_vi_2)#512*224*224
        x_vi_real=self.sigmod((x_vi_map * x_vi) - x_vi_map + 1)
        x_vi_real=self.norm_1_6(x_vi_real)+x_vi
        x_vi_real=self.norm_1_7(x_vi_real)
        # x_vi_real=(x_vi_real - torch.min(x_vi_real)) / (torch.max(x_vi_real) - torch.min(x_vi_real) + 1e-6)
        # x_vi_real = x_vi_real * 255
        # x_vi_3=self.untransformer(x_vi_2)
        # w = 0.95
        # maxV1 = 0.80
        # V1 = torch.clamp(V1 * w, max=maxV1)
        # A=self.getV1(x_vi,V1)
        # # print(m.shape)
        # # print(V1.shape)
        # J = (x_vi - V1) / (1 - V1 / A)
        # print(J.shape)
        # x_out=self.untransformer(J)
        # t=1-V1/A
        return x_vi_real

if __name__ == '__main__':
    vi=torch.randn(1,1,416,416)
    ir=torch.randn(1,1,416,416)
    h = vi.shape[2]
    w = vi.shape[3]
    in_chans=1
    out_chans=1
    text_features=torch.randn(1,77,512)
    SwinFuse_model=SwinFuse(img_size=416,in_chans=in_chans, out_chans=out_chans)
    fudion_model=FusionBlock_res(img_size=416,embed_dim=512,out_chans=1)
    # vi=fudion_model.img_dazing(vi,ir)
    img1_en1,pos1 = SwinFuse_model.self_embed(vi,h,w)
    img2_en1,pos2 = SwinFuse_model.self_embed(ir,h,w)
    img1_en1_daze=fudion_model.img_dazing(img1_en1,img2_en1)

    img1_en2=SwinFuse_model.encoder(img1_en1)
    # img1_en3 = fudion_model.encoder_vi(img1_en2)
    # print(img1_en2.shape)
    img1_texture,sim_vi,fea_cos_vi= fudion_model.encoder_text(img1_en2,text_features)

    img2_en2=SwinFuse_model.encoder(img2_en1)
    # img2_en3 = fudion_model.encoder_ir(img2_en2)
    img2_texture,sim_ir,fea_cos_ir = fudion_model.encoder_text(img2_en2,text_features)

    fusion_emd=fudion_model(img1_en2,img2_en2,img1_texture,img2_texture)
    sim_fusion_vi=fudion_model.loss_cam(fea_cos_vi,fusion_emd)
    sim_fusion_ir=fudion_model.loss_cam(fea_cos_ir,fusion_emd)
    outputs=SwinFuse_model.up_x4(fusion_emd)
    print(outputs.shape)
