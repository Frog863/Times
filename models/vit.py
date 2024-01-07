import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from IPython import embed
import torch.nn.functional as F
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size_h,image_size_w, patch_size_w,  dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = image_size_h,image_size_w
        patch_height, patch_width = image_size_h, patch_size_w
        self.patch_height = patch_height
        self.patch_width = patch_width
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)*channels
        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width
        patch_dim =  patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # embed()
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w c) (p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        # embed()
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b (h w c) (p1 p2) -> b c (h p1) (w p2)', c = channels ,h=image_size_h//image_size_h , p1=image_size_h, p2=patch_width),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches , dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.to_latent = nn.Linear(in_dim, output_dim)

    def forward(self, img):
        # embed()
        x = self.to_patch_embedding(img)
        # embed()
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n )]
        x = self.dropout(x)
        # embed()
        x = self.transformer(x)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # embed()
        x = self.to_latent(x)
        # embed()
        # 将重排后的张量还原为重排前的形状
        x = self.from_patch_embedding(x)
        # embed()
        # x = x.narrow(1, 0, x.size(1)-1)
        return x


# x = torch.randn(16, 32, 1, 1751)
# # x = torch.randn(16, 3, 224, 224)
# remainder =x.size(3) % 4
# if remainder != 0:
#     padding = 4 - remainder
#     pad_dims = (0, padding, 0, 0)
#     x = F.pad(x, pad_dims, mode='constant', value=0)
# patch_w_size = 4
# in_channels = x.size(1)
# embed_dim = x.size(1)*4*x.size(2)
# num_heads =8
# num_layers = 2
# mlp_ratio = 4
# batchsize = x.size(0)
# image_size_h = x.size(2)
# image_size_w = x.size(3)
# model = ViT(image_size_h=image_size_h,image_size_w=image_size_w , patch_size_w=image_size_w//patch_w_size,  dim=x.size(3)//patch_w_size, depth=2, heads=8, mlp_dim=4*x.size(3)//patch_w_size, pool = 'cls', channels = in_channels, dim_head = 64, dropout = 0., emb_dropout = 0.)
#
# out = model(x)
# embed()

