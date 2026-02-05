from typing import Type, Callable, Tuple, Optional, Set, List, Union
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from basicsr.archs.Transformer_Block import DropPath, trunc_normal_init, constant_init, normal_init, LayerNorm
from basicsr.utils.registry import ARCH_REGISTRY
import os


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body1 = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.PixelUnshuffle(2))
        self.body2 = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.PixelUnshuffle(2))

    def forward(self, rgb, h):
        return self.body1(rgb), self.body2(h)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body1 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.PixelShuffle(2))
        self.body2 = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.PixelShuffle(2))

    def forward(self, rgb, h):
        return self.body1(rgb), self.body2(h)


class DWFF(nn.Module):
    def __init__(self,
                 in_channels: int,
                 height: int = 2,
                 reduction: int = 8,
                 bias: bool = True) -> None:
        super(DWFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, d, 1, padding=0, bias=bias),
            nn.GELU()  # nn.LeakyReLU(0.2)
        )

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


class DWFFFusion(nn.Module):
    def __init__(self,
                 in_channels: int,
                 height: int = 2,
                 reduction: int = 8,
                 bias: bool = True) -> None:
        super(DWFFFusion, self).__init__()

        self.rgb = DWFF(in_channels, height, reduction, bias)
        self.h = DWFF(in_channels, height, reduction, bias)

    def forward(self, rgb, h):
        return self.rgb(rgb), self.h(h)


class SpatialAttn(nn.Module):
    def __init__(self, dim: int):
        super(SpatialAttn, self).__init__()

        self.split_c1, self.split_c2, self.split_c3 = int((3 / 8) * dim), int((1 / 8) * dim), int((4 / 8) * dim)
        self.region = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.spatial_1 = nn.Conv2d(self.split_c1, self.split_c1, 5, stride=1, padding=4, groups=self.split_c1,
                                   dilation=2)
        self.spatial_2 = nn.Conv2d(self.split_c3, self.split_c3, 7, stride=1, padding=9, groups=self.split_c3,
                                   dilation=3)
        self.fusion = nn.Conv2d(dim, dim, 1)

        self.gate = nn.SiLU()
        self.proj_value = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
        )
        self.proj_query = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.out = nn.Conv2d(dim, dim, 1)  # nn.Identity()

    def forward(self, x):
        value = self.proj_value(x)
        query = self.proj_query(x)
        query = self.region(query)
        attn = self.gate(
            self.fusion(
                torch.cat([
                    self.spatial_1(query[:, :self.split_c1, :, :]),
                    query[:, self.split_c1:(self.split_c1 + self.split_c2), :, :],
                    self.spatial_2(query[:, (self.split_c1 + self.split_c2):, :, :])
                ], dim=1)
            )
        )
        return self.out(attn * value)


class PGLDE(nn.Module):
    def __init__(self,
                 dim: int = 64,
                 mlp_ratio: float = 2.,
                 drop_path: float = 0.,
                 act_layer: Type[nn.Module] = nn.GELU,
                 norm: Type[nn.Module] = nn.BatchNorm2d) -> None:
        super().__init__()
        self.norm1 = norm(dim)
        self.attn = SpatialAttn(dim=dim)  # MBConv(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm(dim)
        self.mlp = Mixer(dim, mlp_ratio, act_layer)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),

            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.GELU(),
            # CALayer(in_channels, 4),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU(),

            ChannelAttention(out_channels, 4),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))
        )

    def forward(self, x):
        return self.main_path(x) + x


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.GELU(),  # nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.SiLU(True)  # nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=num_feat),
            nn.GELU(),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=num_feat),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x) + x


class CABGroup(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16, num=4):
        super(CABGroup, self).__init__()

        self.group = nn.Sequential(*[CAB(num_feat, squeeze_factor) for _ in range(num)],
                                   nn.Conv2d(num_feat, num_feat, 3, 1, 1))

    def forward(self, x):
        return x + self.group(x)


class ChannelAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, norm: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        # self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1)

        self.q = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        )
        # self.normq, self.normk, self.normv = norm(dim), norm(dim), norm(dim)
        self.norm_q, self.norm_kv = norm(dim), norm(dim)

    def forward(self, q_, k_, v_):  # b,c,h,w -> b,(c n), h,w -> b, n, c, (h, w)
        # q, k, v = self.normq(q_), self.normk(k_), self.normv(v_)
        q, k, v = self.norm_q(q_), self.norm_kv(k_), self.norm_kv(v_)
        b, c, h, w = q.shape
        q, k, v = self.q(q), self.k(k), self.v(v)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj(out)
        return out, k_, v_


class Mixer(nn.Module):
    def __init__(self,
                 in_features: int = 64,
                 mlp_ratio: float = 4,
                 act_layer: Type[nn.Module] = nn.GELU,
                 drop: float = 0.) -> None:
        super(Mixer, self).__init__()
        hidden_features = int(in_features * mlp_ratio)  # hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.ca = nn.Identity()  # BiAttn(in_features)  # eca_layer(in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.ca(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim: int = 64,
                 num_heads: int = 4,
                 bias: bool = True,
                 mlp_ratio: float = 2.,
                 highblocks: int = 2,
                 drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: Type[nn.Module] = nn.GELU,
                 norm: Type[nn.Module] = nn.BatchNorm2d):
        super(Block, self).__init__()

        self.PGLDE = PGLDE(dim, mlp_ratio, drop_path, act_layer, norm)  # nn.Sequential(*[ECAI(dim, mlp_ratio, drop_path, act_layer, norm) for _ in range(highblocks)])
        self.GCBPI = ChannelAttn(dim, num_heads, bias)
        self.PGLGA = ChannelAttn(dim, num_heads, bias)
        self.mlp = Mixer(dim, mlp_ratio, act_layer, drop)
        self.gamma = nn.Parameter(1e-2 * torch.ones((dim)), requires_grad=True)  # noqa
        self.mlp_norm = norm(dim)

    def forward(self, rgb, h):
        q_, k, v = self.GCBPI(rgb, h, h)  # * self.gamma.unsqueeze(-1).unsqueeze(-1) + rgb
        q_ = self.PGLDE(q_ * self.gamma.unsqueeze(-1).unsqueeze(-1) + rgb)
        kv_, _, _ = self.PGLGA(v, q_, q_)
        mlp_in = kv_ + v
        kv_ = self.mlp(self.mlp_norm(mlp_in)) + mlp_in
        return q_, kv_


@ARCH_REGISTRY.register()
class GBPGNet(nn.Module):
    def __init__(self,
                 depths: List = [1, 3, 3, 4],  # Encoder: 4, 6, 6 | Latent: 8 | Decoder: 6, 6, 4   # noqa

                 inc: int = 3,
                 ouc: int = 3,
                 embed_dims: int = 32,
                 num_heads: List = [1, 2, 4, 8],  # noqa  [64, 128, 256, 512] -> [16, 32, 64, 128]
                 bias: bool = True,
                 highblocks: int = [2, 2, 2, 2],  # noqa
                 num_features: int = 64,
                 scale: int = 1,
                 mlp_ratios: float = 2.,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 act_layer: Type[nn.Module] = nn.GELU,
                 norm: Type[nn.Module] = LayerNorm,  # nn.BatchNorm2d,
                 use_mean: bool = False,
                 # load: bool = False,
                 load_path: str = '') -> None:
        super(GBPGNet, self).__init__()

        self.upscale = scale
        self.num_stages = len(depths)
        self.depths = depths
        self.mean = torch.Tensor((0.4488, 0.4371, 0.4040) if use_mean else (0., 0., 0.)).view(1, 3, 1, 1)
        self.use_mean = use_mean

        self.rgb = nn.Conv2d(inc, embed_dims, 3, 1, 1)
        self.h = nn.Conv2d(1, embed_dims, 3, 1, 1)
        self.tail = nn.Conv2d(embed_dims, ouc, 3, 1, 1)

        # Encoder
        self.encoder0 = nn.ModuleList([
            Block(
                dim=embed_dims,
                num_heads=num_heads[0],
                bias=bias,
                mlp_ratio=mlp_ratios,
                highblocks=highblocks[0],
                drop=drop_rate,
                drop_path=drop_path_rate,
                act_layer=act_layer,
                norm=norm
            ) for _ in range(depths[0])
        ])
        self.down0 = Downsample(embed_dims)  # embed_dims * 2 = 128

        self.encoder1 = nn.ModuleList([
            Block(
                dim=embed_dims * 2,
                num_heads=num_heads[1],
                bias=bias,
                mlp_ratio=mlp_ratios,
                highblocks=highblocks[1],
                drop=drop_rate,
                drop_path=drop_path_rate,
                act_layer=act_layer,
                norm=norm
            ) for _ in range(depths[1])
        ])
        self.down1 = Downsample(embed_dims * 2)  # embed_dims * 4 = 256

        self.encoder2 = nn.ModuleList([
            Block(
                dim=embed_dims * 4,
                num_heads=num_heads[2],
                bias=bias,
                mlp_ratio=mlp_ratios,
                highblocks=highblocks[2],
                drop=drop_rate,
                drop_path=drop_path_rate,
                act_layer=act_layer,
                norm=norm
            ) for _ in range(depths[2])
        ])
        self.down2 = Downsample(embed_dims * 4)  # embed_dims * 8 = 512

        # Latent
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dims * 8,
                num_heads=num_heads[3],
                bias=bias,
                mlp_ratio=mlp_ratios,
                highblocks=highblocks[3],
                drop=drop_rate,
                drop_path=drop_path_rate,
                act_layer=act_layer,
                norm=norm
            ) for _ in range(depths[3])
        ])

        # Decoder
        self.up2 = Upsample(embed_dims * 8)  # embed_dims * 4 = 256
        self.decoder2 = nn.ModuleList([
            Block(
                dim=embed_dims * 4,
                num_heads=num_heads[2],
                bias=bias,
                mlp_ratio=mlp_ratios,
                highblocks=highblocks[2],
                drop=drop_rate,
                drop_path=drop_path_rate,
                act_layer=act_layer,
                norm=norm
            ) for _ in range(depths[2])
        ])

        self.up1 = Upsample(embed_dims * 4)  # embed_dims * 2 = 128
        self.decoder1 = nn.ModuleList([
            Block(
                dim=embed_dims * 2,
                num_heads=num_heads[1],
                bias=bias,
                mlp_ratio=mlp_ratios,
                highblocks=highblocks[1],
                drop=drop_rate,
                drop_path=drop_path_rate,
                act_layer=act_layer,
                norm=norm
            ) for _ in range(depths[1])
        ])

        self.up0 = Upsample(embed_dims * 2)  # embed_dims * 1 = 64
        self.decoder0 = nn.ModuleList([
            Block(
                dim=embed_dims,
                num_heads=num_heads[0],
                bias=bias,
                mlp_ratio=mlp_ratios,
                highblocks=highblocks[0],
                drop=drop_rate,
                drop_path=drop_path_rate,
                act_layer=act_layer,
                norm=norm
            ) for _ in range(depths[0])
        ])

        # Refinement, refer to Restormer/PromptIR
        self.refine = nn.ModuleList([
            Block(
                dim=embed_dims,
                num_heads=num_heads[0],
                bias=bias,
                mlp_ratio=mlp_ratios,
                highblocks=highblocks[0],
                drop=drop_rate,
                drop_path=drop_path_rate,
                act_layer=act_layer,
                norm=norm
            ) for _ in range(depths[0])
        ])

        # Encoder-Decoder-Fusion
        self.fusion2 = DWFFFusion(embed_dims * 4)  # Fusion(embed_dims * 8, embed_dims * 4)  #
        self.fusion1 = DWFFFusion(embed_dims * 2)  # Fusion(embed_dims * 4, embed_dims * 2)  #
        self.fusion0 = DWFFFusion(embed_dims * 1)  # Fusion(embed_dims * 2, embed_dims * 1)  #

        # self.apply(self._init_weights)  # noqa
        # Parameters init
        if load_path != '':
            path = os.path.join(
                os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir)), load_path)
            self._load(path, 'state_dict')

    def _init_weights(self, m):
        # print('init weights')
        # for m in self.modules():  self._init_weights()  # noqa
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=.02, bias=0.)
        elif isinstance(m, (nn.LayerNorm, LayerNorm)):
            constant_init(m, val=1.0, bias=0.)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def _load(self, path, key='params', delete_module=False, delete_str='module.'):
        from collections import OrderedDict

        def delete_state_module(weights, delete_str='module.'):
            weights_dict = {}
            for k, v in weights.items():
                # new_k = k.replace('module.', '') if 'module' in k else k
                new_k = k.replace(delete_str, '') if delete_str in k else k
                weights_dict[new_k] = v
            return weights_dict

        ckpt = torch.load(path, map_location=lambda storage, loc: storage)  # ['state_dict']
        ckpt = ckpt.get(key, ckpt)

        if delete_module:
            ckpt = delete_state_module(ckpt, delete_str)

        overlap = OrderedDict()
        for key, value in ckpt.items():
            if key in self.state_dict().keys() and ckpt[key].shape == self.state_dict()[key].shape:
                overlap[key] = value
            else:
                try:
                    print(f'Failed load: ckpt: {key}-{ckpt[key].shape} | model: {key}-{self.state_dict()[key].shape}')
                except Exception as e:
                    print(f'ckpt_len > model_state_len. Failed load: ckpt: {key}-{ckpt[key].shape}')

        print(f'{(len(overlap) * 1.0 / len(ckpt) * 100):.4f}% weights is loaded!', end='\t')
        print(f'{(len(overlap) * 1.0 / len(self.state_dict()) * 100):.4f}% params is inited!')

        try:
            self.load_state_dict(overlap, strict=False)
            print(f'Loading weights from {path}.')
        except RuntimeError as e:
            print(f'RuntimeError: {e}')

    def check_image_size(self, img, scale=8):  # noqa, because of 8x downsample
        _, _, h, w = img.shape
        mod_pad_h = (scale - h % scale) % scale
        mod_pad_w = (scale - w % scale) % scale
        x = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, rgb, h):
        b_, c_, h_, w_ = rgb.shape
        rgb = self.check_image_size(rgb, scale=8)
        h = self.check_image_size(h, scale=8)
        input_img = rgb

        # Head
        rgb = self.rgb(rgb)  # [B, 64, 256, 256]  # noqa
        h = self.h(h)

        # Encoder
        for i, blk in enumerate(self.encoder0):  # [B, 64, 256, 256] -> [B, 64, 256, 256]
            if i == 0:
                rgb0, h0 = blk(rgb, h)
            else:
                rgb0, h0 = blk(rgb0, h0)
        down_rgb0, down_h0 = self.down0(rgb0, h0)  # [B, 64, 256, 256] -> [B, 128, 128, 128]

        for i, blk in enumerate(self.encoder1):  # [B, 128, 128, 128] -> [B, 128, 128, 128]
            if i == 0:
                rgb1, h1 = blk(down_rgb0, down_h0)
            else:
                rgb1, h1 = blk(rgb1, h1)
        down_rgb1, down_h1 = self.down1(rgb1, h1)  # [B, 128, 128, 128] -> [B, 256, 64, 64]

        for i, blk in enumerate(self.encoder2):  # [B, 256, 64, 64] -> [B, 256, 64, 64]
            if i == 0:
                rgb2, h2 = blk(down_rgb1, down_h1)
            else:
                rgb2, h2 = blk(rgb2, h2)
        down_rgb2, down_h2 = self.down2(rgb2, h2)  # [B, 256, 64, 64] -> [B, 512, 32, 32]

        # Latent
        for i, blk in enumerate(self.bottleneck):  # [B, 512, 32, 32] -> [B, 512, 32, 32]
            if i == 0:
                latent_rgb2, latent_h2 = blk(down_rgb2, down_h2)
            else:
                latent_rgb2, latent_h2 = blk(latent_rgb2, latent_h2)

        # Decoder
        up_rgb2, up_h2 = self.up2(latent_rgb2, latent_h2)  # [B, 512, 32, 32] -> [B, 256, 64, 64]
        fusion_rgb2, fusion_h2 = self.fusion2([up_rgb2, rgb2],
                                              [up_h2, h2])  # [B, 256, 64, 64] + [B, 256, 64, 64] -> [B, 256, 64, 64]
        for i, blk in enumerate(self.decoder2):  # [B, 256, 64, 64] -> [B, 256, 64, 64]
            if i == 0:
                de_rgb2, de_h2 = blk(fusion_rgb2, fusion_h2)
            else:
                de_rgb2, de_h2 = blk(de_rgb2, de_h2)

        up_rgb1, up_h1 = self.up1(de_rgb2, de_h2)  # [B, 256, 64, 64] -> [B, 128, 128, 128]
        fusion_rgb1, fusion_h1 = self.fusion1([up_rgb1, rgb1], [up_h1,
                                                                h1])  # [B, 128, 128, 128] + [B, 128, 128, 128] -> [B, 128, 128, 128]
        for i, blk in enumerate(self.decoder1):  # [B, 128, 128, 128] -> [B, 128, 128, 128]
            if i == 0:
                de_rgb1, de_h1 = blk(fusion_rgb1, fusion_h1)
            else:
                de_rgb1, de_h1 = blk(de_rgb1, de_h1)

        up_rgb0, up_h0 = self.up0(de_rgb1, de_h1)  # [B, 128, 128, 128] -> [B, 64, 256, 256]
        fusion_rgb0, fusion_h0 = self.fusion0([up_rgb0, rgb0],
                                              [up_h0, h0])  # [B, 64, 256, 256] + [B, 64, 256, 256] -> [B, 64, 256, 256]
        for i, blk in enumerate(self.decoder0):  # [B, 64, 256, 256] -> [B, 64, 256, 256]
            if i == 0:
                de_rgb0, de_h0 = blk(fusion_rgb0, fusion_h0)
            else:
                de_rgb0, de_h0 = blk(de_rgb0, de_h0)

        # Refinement
        for i, blk in enumerate(self.decoder0):  # [B, 64, 256, 256] -> [B, 64, 256, 256]
            de_rgb0, de_h0 = blk(de_rgb0, de_h0)

        refine = self.tail(de_h0) + input_img  # + identity
        return refine[:, :, :h_, :w_]


if __name__ == '__main__':
    # This section is for development and testing only.
    # It demonstrates model instantiation and basic forward pass.

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration
    model_config = {
        "depths": [1, 3, 3, 4],
        "embed_dims": 32,
    }

    # Create model
    model = GBPGNet(**model_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params / 1e6:.2f}M parameters")

    # Test forward pass
    h, w = 512, 512
    rgb_input = torch.randn(1, 3, h, w).to(device)
    hue_input = torch.randn(1, 1, h, w).to(device)

    for k, v in model.state_dict().items():
        print(k)

    # with torch.no_grad():
    #     output = model(rgb_input, hue_input)
    #     print(f"Output shape: {output.shape}")
    #
    # # Optional: Compute FLOPs and parameters if metrics module is available
    # try:
    #     from basicsr.metrics.ntire.model_summary import get_model_flops
    #
    #     input_dim = (3, h, w)
    #     flops = get_model_flops(model, input_dim, False) / 1e9
    #     num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    #
    #     print(f"FLOPs: {flops:.4f} G")
    #     print(f"Parameters: {num_params:.4f} M")
    # except ImportError:
    #     print("Metrics module not available. Skipping FLOPs computation.")
