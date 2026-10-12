# -*- coding: utf-8 -*-
"""双窗口 E/S/B 网络与共享局部塔片段活性分支。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SPECIES = 19
ESB_COLS = ["Score_E", "Score_S", "Score_B"]


class DilatedBlock(nn.Module):
    def __init__(self, ch, dilation, k=5):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation)
        self.n1 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(min(8, ch), ch)

    def forward(self, x):
        r = x
        x = F.relu(self.n1(self.conv1(x)))
        x = self.n2(self.conv2(x))
        return F.relu(x + r)


class LocalStem(nn.Module):
    """400 bp local tower; returns pooled features and per-position features."""

    def __init__(self, ch=160, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, ch, 9, padding=4),
            nn.GroupNorm(min(8, ch), ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([DilatedBlock(ch, 2 ** i) for i in range(blocks)])
        self.mask_head = nn.Linear(ch, 5)

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        avg = h.mean(dim=2)
        mx = h.max(dim=2).values
        return torch.cat([avg, mx], dim=1), h


class ContextStem(nn.Module):
    """2000 bp upstream context tower (dilated conv encoder, stride-2 stem)."""

    def __init__(self, ch=192, blocks=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, ch // 2, 15, stride=2, padding=7),
            nn.GroupNorm(min(8, ch // 2), ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch // 2, ch, 15, stride=2, padding=7),
            nn.GroupNorm(min(8, ch), ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([DilatedBlock(ch, 2 ** i) for i in range(blocks)])

    def encode(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return h

    def forward(self, x):
        h = self.encode(x)
        avg = h.mean(dim=2)
        mx = h.max(dim=2).values
        return torch.cat([avg, mx], dim=1)


class PromoterAtlasNet(nn.Module):
    def __init__(self, n_tf=0, n_omics=0, n_dist=3, local_ch=160, ctx_ch=192, ctx_blocks=8,
                 local_blocks=6, tf_layers=0, ctx10k=False, ctx10k_ch=128, n_profile=0):
        super().__init__()
        self.local = LocalStem(ch=local_ch, blocks=local_blocks)
        self.context = ContextStem(ch=ctx_ch, blocks=ctx_blocks)
        self.tf_layers = tf_layers
        self.ctx_ch = ctx_ch
        if tf_layers > 0:
            d = ctx_ch
            self.ctx_pos = nn.Parameter(torch.zeros(1, 500, d))
            self.ctx_proj = nn.Identity()
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=d * 4,
                                               dropout=0.1, batch_first=True, norm_first=True)
            self.ctx_transformer = nn.TransformerEncoder(layer, num_layers=tf_layers)
            ctx_dim = 2 * d
        else:
            ctx_dim = 2 * ctx_ch
        self.ctx10k = ctx10k
        if ctx10k:
            self.context10k = ContextStem(ch=ctx10k_ch, blocks=6)
            extra = 2 * ctx10k_ch
        else:
            extra = 0
        self.species_emb = nn.Embedding(N_SPECIES, 32)
        dim = local_ch * 2 + ctx_dim + extra + 32
        self.fuse = nn.Sequential(
            nn.Linear(dim, 512), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )
        self.head_esb = nn.Linear(512, 3)
        self.head_ortho = nn.Linear(512, 1)
        self.head_omics = nn.Linear(512, n_omics) if n_omics else None
        self.head_tf = nn.Linear(512, n_tf) if n_tf else None
        self.head_dist = nn.Linear(512, n_dist) if n_dist else None
        self.head_mpra = nn.Linear(512, 1)
        self.head_species = nn.Linear(local_ch * 2 + ctx_dim, N_SPECIES)
        self.head_profile = nn.Linear(512, n_profile) if n_profile else None

    def forward(self, x400, x400_masked, x2000, species, x10000=None):
        def oh(x):
            return F.one_hot(x.clamp(0, 4).long(), num_classes=5)[..., :4].float().permute(0, 2, 1).contiguous()
        B = x400.shape[0]
        if x400_masked is not None:
            both, both_pos = self.local(oh(torch.cat([x400, x400_masked], dim=0)))
            a = both[:B]
            pos_m = both_pos[B:]
        else:
            a, _ = self.local(oh(x400))
            pos_m = None
        h = self.context.encode(oh(x2000))
        if self.tf_layers > 0:
            ht = h.permute(0, 2, 1)
            ht = ht + self.ctx_pos[:, : ht.size(1)]
            ht = self.ctx_transformer(ht)
            b = torch.cat([ht.mean(dim=1), ht.max(dim=1).values], dim=1)
        else:
            b = torch.cat([h.mean(dim=2), h.max(dim=2).values], dim=1)
        seq_feats = [a, b]
        if self.ctx10k:
            x10 = x10000 if x10000 is not None else torch.full(
                (B, 10000), 4, dtype=torch.long, device=x400.device)
            seq_feats.append(self.context10k(oh(x10)))
        s = self.species_emb(species.clamp(0, N_SPECIES - 1))
        z = self.fuse(torch.cat([s] + seq_feats, dim=1))
        out = {"esb": self.head_esb(z),
               "ortho": self.head_ortho(z).squeeze(-1),
               "dist": self.head_dist(z) if self.head_dist is not None else None,
               "mpra": self.head_mpra(z).squeeze(-1),
               "species": self.head_species(torch.cat([a, b], dim=1))}
        if pos_m is not None:
            out["mask"] = self.local.mask_head(pos_m.permute(0, 2, 1))
        if self.head_omics is not None:
            out["omics"] = self.head_omics(z)
        if self.head_tf is not None:
            out["tf"] = self.head_tf(z)
        if self.head_profile is not None:
            out["profile"] = self.head_profile(z)
        return out


class JointDualESBModel(nn.Module):
    """Paired-window backbone plus an auxiliary fragment-activity head.

    forward_esb is the public inference path. forward_fragment is an auxiliary research
    diagnostic trained on the shared local tower; it reads a single measured DNA fragment.
    """

    def __init__(self, arch: dict, n_tf=0, n_omics=0, n_dist=3,
                 activity_hidden=256, activity_dropout=0.1, n_profile=0):
        super().__init__()
        self.base = PromoterAtlasNet(n_tf=n_tf, n_omics=n_omics, n_dist=n_dist, n_profile=n_profile, **arch)
        self.local_ch = int(arch["local_ch"])
        self.head_activity = nn.Sequential(
            nn.Linear(2 * self.local_ch, activity_hidden), nn.ReLU(inplace=True),
            nn.Dropout(activity_dropout), nn.Linear(activity_hidden, 1))

    @classmethod
    def from_checkpoint(cls, obj: dict) -> "JointDualESBModel":
        a = obj["arch"]
        m = cls(arch={k: a[k] for k in ("local_ch", "ctx_ch", "ctx_blocks", "local_blocks",
                                        "tf_layers", "ctx10k")},
                n_tf=obj.get("n_tf", 0), n_omics=obj.get("n_omics", 0), n_dist=obj.get("n_dist", 3),
                activity_hidden=obj.get("activity_hidden", 256),
                activity_dropout=obj.get("activity_dropout", 0.1),
                n_profile=obj.get("n_profile", 0))
        return m

    @classmethod
    def from_base(cls, base: PromoterAtlasNet, local_ch: int, activity_hidden: int = 256,
                  activity_dropout: float = 0.1) -> "JointDualESBModel":
        """Build only the activity head around an already-loaded backbone.

        Warm start builds the backbone and loads its weights first, then constructs the
        activity head; keeping the two steps separate lets the caller set the head seed
        (torch.manual_seed) after the backbone exists and before the head is created, so the
        head initialisation is reproducible and independent of backbone RNG use.
        """
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.base = base
        obj.local_ch = int(local_ch)
        obj.head_activity = nn.Sequential(
            nn.Linear(2 * obj.local_ch, activity_hidden), nn.ReLU(inplace=True),
            nn.Dropout(activity_dropout), nn.Linear(activity_hidden, 1))
        return obj

    def forward_esb(self, x400, x2000, species, x400_masked=None) -> dict:
        return self.base(x400, x400_masked, x2000, species)

    def forward_fragment(self, codes: torch.Tensor) -> torch.Tensor:
        """Standardized fragment activity; codes (B, L) long 0..4."""
        local = self.base.local
        valid = codes < 4
        x = F.one_hot(codes.clamp(0, 4).long(), 5)[..., :4].float().permute(0, 2, 1).contiguous()
        m = valid.unsqueeze(1).to(x.dtype)
        stem = local.stem
        h = F.relu(stem[1](stem[0](x))) * m
        for blk in local.blocks:
            r = h
            y = F.relu(blk.n1(blk.conv1(h)))
            y = blk.n2(blk.conv2(y))
            h = F.relu(y + r) * m
        denom = m.sum(dim=2).clamp_min(1.0)
        avg = (h * m).sum(dim=2) / denom
        any_valid = valid.any(dim=1, keepdim=True)
        neg = torch.full_like(h, -1e30)
        mx = torch.where(valid.unsqueeze(1), h, neg).max(dim=2).values
        mx = torch.where(any_valid, mx, torch.zeros_like(mx))
        pooled = torch.cat([avg, mx], dim=1)
        return self.head_activity(pooled).squeeze(-1)
