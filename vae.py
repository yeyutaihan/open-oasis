"""
Adapted from https://github.com/endernewton/tokenizer/blob/vae-synced/models_vit.py
"""
import numpy as np
import math
import functools
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Mlp
from timm.layers.helpers import to_2tuple
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, dim=1):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        if dim == 1:
            self.dims = [1, 2, 3]
        elif dim == 2:
            self.dims = [1, 2]
        else:
            raise NotImplementedError
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.mean(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=self.dims,
                )
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=self.dims,
                )

    def nll(self, sample):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.mean(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=self.dims,
        )

    def mode(self):
        return self.mean

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        frame_height,
        frame_width,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal

        rotary_freqs = RotaryEmbedding(
            dim=head_dim // 4,
            freqs_for="pixel", 
            max_freq=frame_height*frame_width,
        ).get_axial_freqs(frame_height, frame_width)
        self.register_buffer("rotary_freqs", rotary_freqs, persistent=False)

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.frame_height * self.frame_width

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        if self.rotary_freqs is not None:
            q = rearrange(q, "b h (H W) d -> b h H W d", H=self.frame_height, W=self.frame_width)
            k = rearrange(k, "b h (H W) d -> b h H W d", H=self.frame_height, W=self.frame_width)
            q = apply_rotary_emb(self.rotary_freqs, q)
            k = apply_rotary_emb(self.rotary_freqs, k)
            q = rearrange(q, "b h H W d -> b h (H W) d")
            k = rearrange(k, "b h H W d -> b h (H W) d")

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop,
            is_causal=self.is_causal,
        )
        x = attn.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        frame_height,
        frame_width,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        attn_causal=False,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads,
            frame_height,
            frame_width,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_causal=attn_causal,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_height=256,
        img_width=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = (img_height, img_width)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, random_sample=False):
        B, C, H, W = x.shape
        assert random_sample or (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        latent_dim,
        input_height=256,
        input_width=256,
        patch_size=16,
        enc_dim=768,
        enc_depth=6,
        enc_heads=12,
        dec_dim=768,
        dec_depth=6,
        dec_heads=12,
        mlp_ratio=4.0,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        use_variational=True,
        **kwargs,
    ):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.patch_size = patch_size
        self.seq_h = input_height // patch_size
        self.seq_w = input_width // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.patch_dim = 3 * patch_size**2

        self.latent_dim = latent_dim
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim

        # patch
        self.patch_embed = PatchEmbed(input_height, input_width, patch_size, 3, enc_dim)

        # encoder
        self.encoder = nn.ModuleList(
            [
                AttentionBlock(
                    enc_dim,
                    enc_heads,
                    self.seq_h,
                    self.seq_w,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(enc_depth)
            ]
        )
        self.enc_norm = norm_layer(enc_dim)

        # bottleneck
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        self.quant_conv = nn.Linear(enc_dim, mult * latent_dim)
        self.post_quant_conv = nn.Linear(latent_dim, dec_dim)

        # decoder
        self.decoder = nn.ModuleList(
            [
                AttentionBlock(
                    dec_dim,
                    dec_heads,
                    self.seq_h,
                    self.seq_w,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_dim)
        self.predictor = nn.Linear(dec_dim, self.patch_dim)  # decoder to patch

        # initialize this weight first
        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        # patchify
        bsz, _, h, w = x.shape
        x = x.reshape(
            bsz,
            3,
            self.seq_h,
            self.patch_size,
            self.seq_w,
            self.patch_size,
        ).permute(
            [0, 1, 3, 5, 2, 4]
        )  # [b, c, h, p, w, p] --> [b, c, p, p, h, w]
        x = x.reshape(
            bsz, self.patch_dim, self.seq_h, self.seq_w
        )  # --> [b, cxpxp, h, w]
        x = x.permute([0, 2, 3, 1]).reshape(
            bsz, self.seq_len, self.patch_dim
        )  # --> [b, hxw, cxpxp]
        return x

    def unpatchify(self, x):
        bsz = x.shape[0]
        # unpatchify
        x = x.reshape(bsz, self.seq_h, self.seq_w, self.patch_dim).permute(
            [0, 3, 1, 2]
        )  # [b, h, w, cxpxp] --> [b, cxpxp, h, w]
        x = x.reshape(
            bsz,
            3,
            self.patch_size,
            self.patch_size,
            self.seq_h,
            self.seq_w,
        ).permute(
            [0, 1, 4, 2, 5, 3]
        )  # [b, c, p, p, h, w] --> [b, c, h, p, w, p]
        x = x.reshape(
            bsz,
            3,
            self.input_height,
            self.input_width,
        )  # [b, c, hxp, wxp]
        return x

    def encode(self, x):
        # patchify
        x = self.patch_embed(x)

        # encoder
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)

        # bottleneck
        moments = self.quant_conv(x)
        if not self.use_variational:
            moments = torch.cat((moments, torch.zeros_like(moments)), 2)
        posterior = DiagonalGaussianDistribution(
            moments, deterministic=(not self.use_variational), dim=2
        )
        return posterior

    def decode(self, z):
        # bottleneck
        z = self.post_quant_conv(z)

        # decoder
        for blk in self.decoder:
            z = blk(z)
        z = self.dec_norm(z)

        # predictor
        z = self.predictor(z)

        # unpatchify
        dec = self.unpatchify(z)
        return dec

    def autoencode(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if self.use_variational and sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior, z

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def synced_step(self, inputs, labels, ratio, split="train"):
        rec, post, latent = self.autoencode(inputs)
        loss, log_dict = self.loss(
            inputs,
            rec,
            post,
            latent,
            0,
            ratio,
            last_layer=self.get_last_layer(),
            split=split,
        )
        d_loss, log_dict_2 = self.loss(
            inputs,
            rec,
            post,
            latent,
            1,
            ratio,
            last_layer=self.get_last_layer(),
            split=split,
        )
        log_dict.update(log_dict_2)
        return loss, d_loss, log_dict

    def disc_loss(self, inputs, rec, post, latent, ratio=1.0, split="train"):
        d_loss, log_dict = self.loss(
            inputs,
            rec,
            post,
            latent,
            1,
            ratio,
            last_layer=self.get_last_layer(),
            split=split,
        )
        return d_loss, log_dict

    def gen_loss(self, inputs, rec, post, latent, ratio=0.0, split="train"):
        loss, log_dict = self.loss(
            inputs,
            rec,
            post,
            latent,
            0,
            ratio,
            last_layer=self.get_last_layer(),
            split=split,
        )
        return loss, log_dict

    def forward(self, inputs, labels, split="train"):
        rec, post, latent = self.autoencode(inputs)
        return rec, post, latent

    def configure_optimizers(
        self,
        learning_rate,
        beta1=0.5,
        beta2=0.9,
        weight_decay=0.0,
    ):
        # remove parameters in loss
        param_loss_names = ["loss." + n for n, p in self.loss.named_parameters()]
        param_ae = [p for n, p in self.named_parameters() if n not in param_loss_names]
        opt_ae = torch.optim.AdamW(
            param_ae,
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
        opt_disc = torch.optim.AdamW(
            self.loss.discriminator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
        return opt_ae, opt_disc

    def get_last_layer(self):
        return self.predictor.weight

    @torch.no_grad()
    def log_images(self, x, only_inputs=False):
        log = dict()
        if not only_inputs:
            xrec, posterior, latent = self.autoencode(x)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log



def ViT_L_20_Shallow_Encoder(**kwargs):
    if "latent_dim" in kwargs:
        latent_dim = kwargs.pop("latent_dim")
    else:
        latent_dim = 16
    return AutoencoderKL(
        latent_dim=latent_dim,
        patch_size=20,
        enc_dim=1024,
        enc_depth=6,
        enc_heads=16,
        dec_dim=1024,
        dec_depth=12,
        dec_heads=16,
        input_height=360,
        input_width=640,
        **kwargs,
    )

VAE_models = {
    "vit-l-20-shallow-encoder": ViT_L_20_Shallow_Encoder,
}
