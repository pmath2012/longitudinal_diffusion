import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    """
    A cross-attention module that uses PyTorch's MultiheadAttention.
    """
    def __init__(self, in_channels, cond_channels, embed_dim, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1) Project x => embed_dim
        self.proj_x = nn.Conv2d(in_channels, embed_dim, 1)
        # 2) Project cond => embed_dim
        self.proj_cond = nn.Conv2d(cond_channels, embed_dim, 1)

        # 3) Use built-in MHA with batch_first=True => [B, N, embed_dim]
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # 4) Project back to in_channels
        self.proj_out = nn.Conv2d(embed_dim, in_channels, 1)

        # optional residual norm
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, in_channels,   H, W]
        cond: [B, cond_channels, H, W]  (assume same H,W)
        """
        B, Cx, H, W = x.shape
        Bc, Cc, Hc, Wc = cond.shape
        assert B == Bc, "Batch mismatch"
        assert H == Hc and W == Wc, "Spatial mismatch (up/down-sample if needed)"

        # 1) Project x, cond => [B, embed_dim, H, W]
        x_ = self.proj_x(x)
        cond_ = self.proj_cond(cond)

        # 2) Flatten => [B, H*W, embed_dim]
        x_ = x_.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        cond_ = cond_.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)

        # 3) Cross-attention => Q from x_, K/V from cond_
        attn_out, _ = self.mha(query=x_, key=cond_, value=cond_)

        # 4) Residual + layer norm
        out = x_ + attn_out
        out = self.layernorm(out)

        # 5) Reshape => [B, embed_dim, H, W]
        out = out.reshape(B, H, W, self.embed_dim).permute(0, 3, 1, 2)

        # 6) Project back => [B, in_channels, H, W], plus final residual
        out = self.proj_out(out)
        return x + out
