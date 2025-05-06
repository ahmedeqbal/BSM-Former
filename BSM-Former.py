import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision.ops import roi_align, nms

# Utility functions
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# U-Net Components
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Transformer Components
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm3p(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm5 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x5, x4, x3, **kwargs):
        return self.fn(self.norm5(x5), self.norm4(x4), self.norm3(x3), **kwargs)

class PreNorm2pm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, mask, **kwargs):
        return self.fn(self.norm(x), mask, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention_global(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_k1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_q2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_k2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim*2, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x5, x4, x3):
        q1 = self.to_q1(x3)
        q1 = rearrange(q1, 'b n (h d) -> b h n d', h=self.heads)
        q2 = self.to_q2(x3)
        q2 = rearrange(q2, 'b n (h d) -> b h n d', h=self.heads)
        k1 = self.to_k1(x4)
        k1 = rearrange(k1, 'b n (h d) -> b h n d', h=self.heads)
        v1 = self.to_v1(x4)
        v1 = rearrange(v1, 'b n (h d) -> b h n d', h=self.heads)
        k2 = self.to_k2(x5)
        k2 = rearrange(k2, 'b n (h d) -> b h n d', h=self.heads)
        v2 = self.to_v2(x5)
        v2 = rearrange(v2, 'b n (h d) -> b h n d', h=self.heads)
        dots1 = torch.matmul(q1, k1.transpose(-1, -2)) * self.scale
        attn1 = self.attend(dots1)
        out1 = torch.matmul(attn1, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        dots2 = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale
        attn2 = self.attend(dots2)
        out2 = torch.matmul(attn2, v2)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out = torch.cat((out1, out2), dim=-1)
        return self.to_out(out)

class Attention_local(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., win_size=16, img_height=256, img_width=256):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.win_size = win_size
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.fixcnn = FixCNN(win_size=win_size//2)
        self.window = Shifted_Windows(img_height, img_width, win_size)
        self.shifty = make_gridsy(win_size)
        self.shiftx = make_gridsx(win_size)

    def forward(self, x, prob):
        b, c, h, w = prob.shape
        _, N, d = x.shape
        log_prob = torch.log2(prob + 1e-10)
        entropy = -1 * torch.sum(prob * log_prob, dim=1)
        x_2d = rearrange(x, 'b (h w) d -> b d h w', h=2 * h, w=2 * w)
        outx_2d = x_2d * 0
        win_cunt = x_2d[:, 0, :, :] * 0
        win_score = self.fixcnn(entropy[:, None, :, :])/(self.win_size//2*self.win_size//2)
        win_score = win_score.view(b, -1)
        window = torch.from_numpy(self.window).to(x.device).float()
        keep_num = min(int(0.7*(2 * h // self.win_size)**2), 50)
        for i in range(b):
            scorei = win_score[i]
            indexi = nms(boxes=window, scores=scorei, iou_threshold=0.2)
            indexi = indexi[:keep_num]
            keep_windowi = window[indexi, :]
            window_batch_indexi = torch.zeros(keep_windowi.shape[0], device=x.device) + i
            index_windowi = torch.cat([window_batch_indexi[:, None], keep_windowi], dim=1)
            window_featurei = roi_align(x_2d, index_windowi, (self.win_size, self.win_size))
            xi = rearrange(window_featurei, 'm d h w -> m (h w) d')
            qkvi = self.to_qkv(xi).chunk(3, dim=-1)
            qi, ki, vi = map(lambda t: rearrange(t, 'm n (h d) -> m h n d', h=self.heads), qkvi)
            dotsi = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale
            attni = self.attend(dotsi)
            outi = torch.matmul(attni, vi)
            outi = rearrange(outi, 'm h n d -> m n (h d)')
            outi = self.to_out(outi)
            outi_2d = rearrange(outi, 'm (h w) d -> m d h w', h=self.win_size)
            m = outi.shape[0]
            for j in range(m):
                sy = int(keep_windowi[j, 1])
                sx = int(keep_windowi[j, 0])
                outx_2d[i, :, sy:sy+self.win_size, sx:sx+self.win_size] += outi_2d[j, :, :, :]
                win_cunt[i, sy:sy+self.win_size, sx:sx+self.win_size] += 1
        outx = rearrange(outx_2d/(win_cunt[:, None, :, :] + 1e-10), 'b d h w -> b (h w) d')
        x = x + outx
        return x

class FixCNN(nn.Module):
    def __init__(self, win_size=16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, win_size, win_size))
    def forward(self, x):
        out = F.conv2d(x, self.weight, bias=None, stride=1, padding=0)
        return out

def Shifted_Windows(height, width, win_size, stride=2):
    shift_y = torch.arange(0, height-win_size+1, stride)
    shift_x = torch.arange(0, width-win_size+1, stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()), axis=1)
    M = shift.shape[0]
    window = shift.reshape(M, 4)
    window[:, 2] = window[:, 0] + win_size-1
    window[:, 3] = window[:, 1] + win_size-1
    return window

def make_gridsx(win_size):
    shift_y = torch.arange(0, win_size, 1)
    shift_x = torch.arange(0, win_size, 1)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    return torch.tensor(shift_x)

def make_gridsy(win_size):
    shift_y = torch.arange(0, win_size, 1)
    shift_x = torch.arange(0, win_size, 1)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    return torch.tensor(shift_y)

class Transformer_global(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm3p(dim, Attention_global(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x5, x4, x):
        for attn, ff in self.layers:
            x = attn(x5, x4, x) + x
            x = ff(x) + x
        return x

class Transformer_local(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128, win_size=16, img_height=256, img_width=256):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2pm(dim, Attention_local(dim, heads=heads, dim_head=dim_head, dropout=dropout, win_size=win_size, img_height=img_height, img_width=img_width)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x, fore_score):
        for attn, ff in self.layers:
            x = attn(x, fore_score)
            x = ff(x) + x
        return x

class GCST(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel * 4

        self.to_patch_embedding_x5 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),
            nn.Linear(in_channels*4, self.dmodel),
        )
        self.to_patch_embedding_x4 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=2, p2=2),
            nn.Linear(in_channels*2*4, self.dmodel),
        )
        self.to_patch_embedding_x3 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer_global(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches)
        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

    def forward(self, x5, x4, x3):
        x5 = self.to_patch_embedding_x5(x5)
        x4 = self.to_patch_embedding_x4(x4)
        x3 = self.to_patch_embedding_x3(x3)
        _, n5, _ = x5.shape
        _, n4, _ = x4.shape
        _, n3, _ = x3.shape
        x5 += self.pos_embedding[:, :n5]
        x4 += self.pos_embedding[:, :n4]
        x3 += self.pos_embedding[:, :n3]
        x5 = self.dropout(x5)
        x4 = self.dropout(x4)
        x3 = self.dropout(x3)
        ax = self.transformer(x5, x4, x3)
        out = self.recover_patch_embedding(ax)
        return out

class BSLT(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, depth=2, n_classes=9, patch_size=1, win_size=16, heads=6, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel * 4

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer_local(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches, win_size, image_height, image_width)
        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

    def forward(self, x1, x2):
        x = self.to_patch_embedding(x1)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        ax = self.transformer(x, x2)
        out = self.recover_patch_embedding(ax)
        return out

# BSM-Transformer Model
class BSMTrans(nn.Module):
    def __init__(self, global_block, local_block, layers, n_channels, n_classes, imgsize, patch_size=2, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.scale = 4

        self.inc = DoubleConv(n_channels, 64 // self.scale)
        self.down1 = Down(64 // self.scale, 128 // self.scale)
        self.down2 = Down(128 // self.scale, 256 // self.scale)
        self.down3 = Down(256 // self.scale, 512 // self.scale)
        factor = 2 if bilinear else 1
        self.down4 = Down(512 // self.scale, 1024 // factor // self.scale)

        self.up4 = Up(1024 // self.scale, 512 // factor // self.scale, bilinear)
        self.up3 = Up(512 // self.scale, 256 // factor // self.scale, bilinear)
        self.up2 = Up(256 // self.scale, 128 // factor // self.scale, bilinear)
        self.up1 = Up(128 // self.scale, 64 // self.scale, bilinear)

        self.softmax = nn.Softmax(dim=1)

        for p in self.parameters():
            p.requires_grad = True

        self.trans_local2 = local_block(128 // self.scale // factor, 128 // self.scale // factor * 2, imgsize // 2, 1, heads=6, patch_size=1, n_classes=n_classes, win_size=16)
        self.trans_global = global_block(256 // factor // self.scale, 256 // factor // self.scale * 2, imgsize // 4, 1, heads=4, patch_size=1)

        self.outc1 = OutConv(64 // self.scale * 4, n_classes)
        self.convl1 = nn.Conv2d(64 // self.scale, 64 // self.scale * 4, kernel_size=1, padding=0, bias=False)
        self.outc2 = OutConv(64 // self.scale * 4, n_classes)
        self.convl2 = nn.Conv2d(128 // factor // self.scale * 2, 64 // self.scale * 4, kernel_size=1, padding=0, bias=False)
        self.outc3 = OutConv(64 // self.scale * 4, n_classes)
        self.convl3 = nn.Conv2d(256 // factor // self.scale * 2, 64 // self.scale * 4, kernel_size=1, padding=0, bias=False)
        self.out = OutConv(3 * 64 // self.scale * 4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d4 = self.up4(x5, x4)
        d3 = self.up3(d4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)

        trans_global = self.trans_global(x5, d4, d3)
        l3 = self.convl3(trans_global)
        pred3 = self.outc3(l3)
        l3_up = l3[:, :, :, :, None].repeat(1, 1, 1, 1, 16)
        l3_up = rearrange(l3_up, 'b c h w (m n) -> b c (h m) (w n)', m=4, n=4)

        pred3_p = self.softmax(pred3)
        trans_local2 = self.trans_local2(d2, pred3_p)
        l2 = self.convl2(trans_local2)
        pred2 = self.outc2(l2)
        l2_up = l2[:, :, :, :, None].repeat(1, 1, 1, 1, 4)
        l2_up = rearrange(l2_up, 'b c h w (m n) -> b c (h m) (w n)', m=2, n=2)

        l1 = self.convl1(d1)
        pred1 = self.outc1(l1)

        predf = torch.cat((l1, l2_up, l3_up), dim=1)
        predf = self.out(predf)

        return predf, pred1, pred2, pred3

def BSMTransformer(pretrained=False, **kwargs):
    model = BSMTrans(GCST, BSLT, [1, 1, 1, 1], **kwargs)
    return model

# Main function to demonstrate the model
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model parameters
    n_channels = 3
    n_classes = 9
    imgsize = 256

    # Initialize model
    model = BSMTransformer(
        n_channels=n_channels,
        n_classes=n_classes,
        imgsize=imgsize,
        bilinear=True
    ).to(device)

    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, n_channels, imgsize, imgsize).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        predf, pred1, pred2, pred3 = model(x)
    
    # Print output shapes
    print(f"Final prediction shape: {predf.shape}")
    print(f"Prediction 1 shape: {pred1.shape}")
    print(f"Prediction 2 shape: {pred2.shape}")
    print(f"Prediction 3 shape: {pred3.shape}")

    # Visualize a sample output (first batch, first class)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input Image (Channel 0)")
    plt.imshow(x[0, 0].cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Output Prediction (Class 0)")
    plt.imshow(predf[0, 0].cpu().numpy(), cmap='jet')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()