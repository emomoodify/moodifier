from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

torch.set_printoptions(threshold=float('inf'))

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # print('input_resolution // patch_size: ', input_resolution, patch_size, width)
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def _global_pool(self, x: torch.Tensor):
        pooled, tokens = x[:, 0], x[:, 1:]
        return pooled, tokens

    def forward(self, x: torch.Tensor):
        # print('in vision transformer, x.shape:', x.shape)   # bs, 3, 224, 224
        x = self.conv1(x)
        # print('in vision transformer, after conv1, x.shape:', x.shape)   # bs, 1024, 16, 16
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        # print('in vision transformer, after reshape and permute, x.shape:', x.shape)    # bs, 256, 1024
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        # print('in vision transformer, after cat, x.shape:', x.shape)            # bs, 257, 1024

        x = x + self.positional_embedding.to(x.dtype)
        # print('in vision transformer, after adding positional embedding, x.shape:', x.shape)    # bs, 257, 1024

        x = self.ln_pre(x)
        # print('in vision transformer, after ln_pre, x.shape:', x.shape)     # bs, 257, 1024

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # print('in vision transformer, after transformer, x.shape:', x.shape)    # 257, bs, 1024

        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        #
        # if self.proj is not None:
        #     x = x @ self.proj

        x = self.ln_post(x)
        # print('in vision transformer, after ln_post, x.shape:', x.shape)    # bs, 257, 1024
        pooled, tokens = self._global_pool(x)
        # print('in vision transformer, after global pool, pooled.shape:', pooled.shape)  # torch.Size([bs, 1024])
        # print('in vision transformer, after global pool, tokens.shape:', tokens.shape)  # torch.Size([bs, 256, 1024])

        if self.proj is not None:
            pooled = pooled @ self.proj
        # print('in vision transformer, after proj, at last, pooled.shape:', pooled.shape)    # bs, 768
        # print('in vision transformer, after proj, at last, tokens.shape:', tokens.shape)    # bs, 576, 1024
        return pooled, tokens


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 load_from_clip: bool
                 ):
        super().__init__()

        self.context_length = 248

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)

        if load_from_clip == False:
            self.positional_embedding = nn.Parameter(torch.empty(248, transformer_width))
            self.positional_embedding_res = nn.Parameter(torch.empty(248, transformer_width))

        else:
            self.positional_embedding = nn.Parameter(torch.empty(77, transformer_width))

        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.eps = 0.1
        self.labels_ot = {}
        self.cache_labels = True

        self.initialize_parameters()
        self.mask1 = torch.zeros([248, 1])
        self.mask1[:20, :] = 1
        self.mask2 = torch.zeros([248, 1])
        self.mask2[20:, :] = 1


    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + (self.positional_embedding.to(x.device) * self.mask1.to(x.device)).type(self.dtype).to(x.device) + (self.positional_embedding_res.to(x.device) * self.mask2.to(x.device)).type(self.dtype).to(x.device)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_text_full(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + (self.positional_embedding.to(x.device) * self.mask1.to(x.device)).type(self.dtype).to(x.device) + (self.positional_embedding_res.to(x.device) * self.mask2.to(x.device)).type(self.dtype).to(x.device)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        return x

    def PCA(self, input_tensor, PCA_dim):
        mean = torch.mean(input_tensor, dim=0)
        X_centered = input_tensor - mean.unsqueeze(0)
        X_centered = X_centered.float()
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        principal_components = Vt.T[:, :PCA_dim]
        X_transformed = torch.mm(X_centered, principal_components)
        X_reversed = torch.mm(X_transformed, principal_components.T)
        X_reversed += mean
        return X_reversed

    def get_ground_truth_ot(self, device, num_logits, M, N) -> torch.Tensor:
        if device not in self.labels_ot:
            labels = torch.eye(num_logits, device=device, dtype=torch.long).repeat(M,N)
            if self.cache_labels:
                self.labels_ot[device] = labels
                self.prev_num_logits_ot = num_logits
        else:
            labels = self.labels_ot[device]
        return labels

    def pairwise_contrastive_loss(self, a, b, device, logit_scale=1.0):
        # a:    all_text_features,
        # b:    l_grouped_v_patch_embed,
        # print('use pairwise_contrastive_loss, text features shape:', a.shape)       # bs, num_text, 768
        # print('use pairwise_contrastive_loss, local image features shape:', b.shape)    # bs, num_text, 768
        batch_size, seq_len, c = a.shape
        labels = torch.eye(seq_len*batch_size, device=device, dtype=torch.float)#.repeat(batch_size, 1)
        # print('use pairwise_contrastive_loss, labels:', labels.shape)

        # logits = torch.einsum('bd,bd->bmn', a, b) * logit_scale
        text_features = a.contiguous().view(seq_len*batch_size, c)
        image_features = b.contiguous().view(seq_len*batch_size, c)
        # print('use pairwise_contrastive_loss, text features shape 2:', text_features.shape)     # bs*num_text, 768
        # print('use pairwise_contrastive_loss, local image features shape 2:', image_features.shape)   # bs*num_text, 768
        logits_per_image = image_features @ text_features.T * logit_scale
        logits_per_text = text_features @ image_features.T * logit_scale
        # print('use pairwise_contrastive_loss, logits_per_image shape:', logits_per_image.shape)     # bs*num_text, bs*num_text
        # print('use pairwise_contrastive_loss, logits_per_text shape:', logits_per_text.shape)       # bs*num_text, bs*num_text

        loss = 1/2 * (F.cross_entropy(logits_per_text, labels) + F.cross_entropy(logits_per_image, labels))
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-3
        for i in range(100):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        # print('in model clip Sinkhorn, u: ', u.shape)                           # (bs*num_gpu)*(bs*num_gpu), 768
        # print('in model clip Sinkhorn, v: ', v.shape)                           # (bs*num_gpu)*(bs*num_gpu), 5
        # print('in model clip Sinkhorn, r: ', r.shape, r.unsqueeze(-1).shape)    # ((bs*num_gpu)*(bs*num_gpu), 768); ((bs*num_gpu)*(bs*num_gpu), 768, 1)
        # print('in model clip Sinkhorn, c: ', c.shape, c.unsqueeze(-2).shape)    # ((bs*num_gpu)*(bs*num_gpu), 5); ((bs*num_gpu)*(bs*num_gpu), 1, 5)
        # print('in model clip Sinkhorn, K: ', K.shape)


        # ((bs*num_gpu)*(bs*num_gpu), 768, 1) * ((bs*num_gpu)*(bs*num_gpu), 1, 5) * ((bs*num_gpu)*(bs*num_gpu), 768, 5)
        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
        # print('in model clip Sinkhorn, T: ', T.shape)                           # (bs*num_gpu)*(bs*num_gpu), 768, 5

        return T

    def forward(self, image, text_long, text_shorts,rank):

        # ================== global image-text matching (like original clip) ==================

        # print('in model clip forward, '
        #       ' image: ', image.shape,              # bs, 3, 336, 336
        #       ' text_long: ', text_long.shape,      # bs, 248
        #       ' text_short: ', text_short.shape)    # bs, 248

        out_losses = {}
        image_features_long, local_image_features_long = self.encode_image(image)
        # print('in model clip forward, image_features_long: ', image_features_long.shape)                # bs, 768
        # print('in model clip forward, local_image_features_long: ', local_image_features_long.shape)    # bs, 576, 1024

        text_features_long = self.encode_text(text_long)

        all_text_features_list = []
        for t in text_shorts:
            all_text_features_list.append(self.encode_text(t))
        all_text_features_vec = torch.stack(all_text_features_list, dim=0).view(image_features_long.shape[0], len(text_shorts),
                                                                      -1)
        # print('all_text_features_vec: ', all_text_features_vec.shape)                       # bs, 5, 768

        text_features_short = all_text_features_vec[:, 0, :] + all_text_features_vec[:, -1, :]
        # print('in model clip forward, text_features_short: ', text_features_short.shape)    # bs, 768

        bs, number_text, c = all_text_features_vec.shape
        # print('in model clip forward, local_image_features_long: ', local_image_features_long.shape)          # bs, 576, 1024
        local_image_features_long = local_image_features_long.contiguous().view(text_features_long.shape[0], 1, -1, c)

        # normalized features
        image_features_long = image_features_long / image_features_long.norm(dim=1, keepdim=True)
        text_features_long = text_features_long / text_features_long.norm(dim=1, keepdim=True)
        text_features_short = text_features_short / text_features_short.norm(dim=1, keepdim=True)

        image_features_short = self.PCA(image_features_long, 32)
        # image_features_short = image_features_long          # temporary fix

        # print('in model clip forward, after PCA, image_features_short: ', image_features_short.shape)           # bs, 768
        # print('in model clip forward, after encode_image, image_features_long: ', image_features_long.shape)    # bs, 768

        image_feat_all_long = torch.cat(torch.distributed.nn.all_gather(image_features_long), dim=0)#gather with grad
        local_feat_all_long = torch.cat(torch.distributed.nn.all_gather(local_image_features_long), dim=0)
        image_features_all_short = torch.cat(torch.distributed.nn.all_gather(image_features_short), dim=0)

        text_feat_all_long = torch.cat(torch.distributed.nn.all_gather(text_features_long), dim=0)
        text_feat_all_short = torch.cat(torch.distributed.nn.all_gather(text_features_short), dim=0)

        # print('in model clip forward, image_features_long: ', image_features_long.shape)        # bs, 768
        # print('in model clip forward, image_feat_all_long: ', image_feat_all_long.shape)        # bs*num_gpu, 768
        # print('in model clip forward, text_features_long: ', text_features_long.shape)          # bs, 768
        # print('in model clip forward, text_feat_all_long: ', text_feat_all_long.shape)          # bs*num_gpu, 768
        # print('in model clip forward, text_features_short: ', text_features_short.shape)        # bs, 768
        # print('in model clip forward, text_feat_all_short: ', text_feat_all_short.shape)        # bs*num_gpu, 768

        sim_i2tl = torch.matmul(image_features_long, text_feat_all_long.T) * self.logit_scale.exp()
        # sim_tl2i = torch.matmul(image_feat_all_long, text_features_long.T)
        # sim_tl2i = sim_tl2i.T
        sim_tl2i = torch.matmul(text_features_long, image_feat_all_long.T) * self.logit_scale.exp()

        sim_i2ts = torch.matmul(image_features_short, text_feat_all_short.T) * self.logit_scale.exp()
        # sim_ts2i = torch.matmul(image_features_all_short, text_features_short.T)
        # sim_ts2i = sim_ts2i.T
        sim_ts2i = torch.matmul(text_features_short, image_features_all_short.T) * self.logit_scale.exp()

        # print('in model clip forward, sim_i2tl: ', sim_i2tl.shape)            # bs, bs*num_gpu
        # print('in model clip forward, sim_tl2i: ', sim_tl2i.shape)            # bs, bs*num_gpu
        # print('in model clip forward, sim_i2ts: ', sim_i2ts.shape)            # bs, bs*num_gpu
        # print('in model clip forward, sim_ts2i: ', sim_ts2i.shape)            # bs, bs*num_gpu

        bs = image.size(0)
        targets = torch.linspace(rank * bs,rank * bs + bs - 1, bs, dtype=torch.long).to(image.device)
        # print('in model clip forward, targets: ', targets.shape)        # bs

        loss_itcl = (
                F.cross_entropy(sim_i2tl, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_tl2i, targets, label_smoothing=0.1)
            ) / 2
        # print('in model clip forward, loss_itcl: ', loss_itcl.shape)

        loss_itcs = (
                F.cross_entropy(sim_i2ts, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_ts2i, targets, label_smoothing=0.1)
            ) / 2
        # print('in model clip forward, loss_itcs: ', loss_itcs.shape)

        out_losses['loss_itcl'] = loss_itcl
        out_losses['loss_itcs'] = loss_itcs

        # ==================  local group fine-grained alignment loss ==================
        bs, num_i, M, c = local_feat_all_long.shape
        b = bs * num_i

        all_text_features_vec = torch.cat(torch.distributed.nn.all_gather(all_text_features_vec), dim=0)
        local_image_features_ = local_feat_all_long.permute(1, 0, 2, 3).contiguous().view(b, M, c)
        all_text_features = all_text_features_vec.contiguous().view(b, number_text, c)

        local_image_features_ = F.normalize(local_image_features_, p=2, dim=-1)
        all_text_features = F.normalize(all_text_features, p=2, dim=-1)

        # print('local_image_features_: ', local_image_features_.shape)     # bs*num_gpu, 768, 768
        # print('all_text_features: ', all_text_features.shape)             # bs*num_gpu, 5, 768

        similarity = torch.einsum('btd,bpd->btp', all_text_features, local_image_features_)
        # print('similarity: ', similarity.shape)                             # bs*num_gpu, 5, 768

        # min_val = torch.min(similarity, dim=-1, keepdim=True).values
        # max_val = torch.max(similarity, dim=-1, keepdim=True).values
        # epsilon = 1e-10
        # normalized_similarity = (similarity - min_val) / (max_val - min_val + epsilon)
        # similarity_threshold = 0.0
        # normalized_similarity = torch.where(normalized_similarity < similarity_threshold,
        #                                     torch.tensor(0.0, device=normalized_similarity.device),
        #                                     normalized_similarity)

        # sum_similarity = torch.sum(similarity, dim=-1, keepdim=True)
        # v_align_weights = similarity / sum_similarity
        v_align_weights = similarity

        # print('v_align_weights: ', v_align_weights.shape)             # bs*num_gpu, 5, 768

        l_grouped_v_patch_embed = torch.einsum('btp,bpd->btd', v_align_weights, local_image_features_)

        # print('l_grouped_v_patch_embed: ', l_grouped_v_patch_embed.shape)  # bs*num_gpu, 5, 768

        l_grouped_v_patch_embed = F.normalize(l_grouped_v_patch_embed, p=2, dim=-1)

        out_losses['finegrained_loss'] = self.pairwise_contrastive_loss(all_text_features,
                                                                        l_grouped_v_patch_embed,
                                                                        image_feat_all_long.device,
                                                                        self.logit_scale.exp())

        # ==================================== local ot loss ====================================

        # bs, M, c = local_image_features_long.shape
        # num_i = 1
        # b = bs * num_i
        # local_image_features = local_image_features_.permute(1, 0, 2, 3).contiguous().view(b, M, c)
        # all_text_features = all_text_features.contiguous().view(bs, number_text, c)

        # print('use ot loss, local_image_features_: ', local_image_features_.shape)    # bs*num_gpu, 768, 768
        # print('use ot loss, all_text_features: ', all_text_features.shape)          # bs*num_gpu, 5, 768

        # (bs, 768, 768) -> (768, bs, 768)
        # (bs, 5, 768) -> (5, bs, 768)
        # (768, bs, 768) * (5, bs, 768) -> (768, 5, bs, bs)

        sim = torch.einsum('mbd,ncd->mnbc', local_image_features_.permute(1, 0, 2),
                           all_text_features.permute(1, 0, 2)).contiguous()
        sim = sim.view(M, number_text, b * bs)
        sim = sim.permute(2, 0, 1)

        # print('use ot loss, sim: ', sim.shape)     # (bs*num_gpu)*(bs*num_gpu), 768, 5

        wdist = 1.0 - sim
        # print('use ot loss, wdist: ', wdist.shape)   # (bs*num_gpu)*(bs*num_gpu), 768, 5

        xx = torch.zeros(b * bs, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy = torch.zeros(b * bs, number_text, dtype=sim.dtype, device=sim.device).fill_(1. / number_text)

        # print('use ot loss, xx: ', xx.shape)        # (bs*num_gpu)*(bs*num_gpu), 768
        # print('use ot loss, yy: ', yy.shape)        # (bs*num_gpu)*(bs*num_gpu), 5

        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            T = self.Sinkhorn(KK, xx, yy)
            # print('use ot loss, T: ', T.shape)       # (bs*num_gpu)*(bs*num_gpu), 768, 5

        try:
            torch.isnan(T).any()
        except None:
            print('There is none value in your tensor, please try to adjust #thre and #eps to align data.')

        sim_op = torch.sum(T * sim, dim=(1, 2))
        # print('use ot loss, sim_op: ', sim_op.shape)    # (bs*num_gpu)*(bs*num_gpu)

        sim_op = sim_op.contiguous().view(b, bs) * self.logit_scale.exp()
        # print('sim_op: ', sim_op.shape)                     # (bs*num_gpu), (bs*num_gpu)

        # print('self.get_ground_truth_ot(device, b, 1, 1).float(): ', self.get_ground_truth_ot(sim.device, b, 1, 1).float())

        out_losses['ot_local_loss'] = F.cross_entropy(sim_op, self.get_ground_truth_ot(sim.device, b, 1, 1).float())

        return out_losses


def gather_only_features(
        features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_features = hvd.allgather(features)
        else:
            with torch.no_grad():
                all_features = hvd.allgather(features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features = list(all_features.chunk(world_size, dim=0))
                gathered_features[rank] = features
                all_features = torch.cat(gathered_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        else:
            gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
            features = features.contiguous()
            torch.distributed.nn.all_gather(gathered_features, features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features[rank] = features
            all_features = torch.cat(gathered_features, dim=0)

    return all_features

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, load_from_clip: bool):
    vit = "visual.proj" in state_dict
    # print('in build_model, vit: ', vit)
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # print('in build_model, input for CLIP, '
    #       'embed_dim: ', embed_dim,                     # 768
    #       ' image_resolution: ', image_resolution,      # 224
    #       ' vision_layers: ', vision_layers,            # 24
    #       ' vision_width: ', vision_width,              # 1024
    #       ' vision_patch_size: ', vision_patch_size,    # 14
    #       ' context_length: ', context_length,          # 77
    #       ' vocab_size: ', vocab_size,                  # 49408
    #       ' transformer_width: ', transformer_width,    # 768
    #       ' transformer_heads: ', transformer_heads,    # 12
    #       ' transformer_layers: ', transformer_layers,  # 12
    #       ' load_from_clip: ', load_from_clip)          # True

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, load_from_clip
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
