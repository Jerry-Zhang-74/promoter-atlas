# -*- coding: utf-8 -*-
# batch52 fresh sequence CNN: stem Conv1d(4,64,9) + 3 residual blocks (dil 1,2,4) + masked mean/max.
import torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, c=64, k=5, d=1):
        super().__init__()
        p = d*(k-1)//2
        self.c1 = nn.Conv1d(c, c, k, padding=p, dilation=d)
        self.c2 = nn.Conv1d(c, c, k, padding=p, dilation=d)
        self.drop = nn.Dropout(0.1)
    def forward(self, h, mask):
        r = h
        h = F.gelu(self.c1(h)); h = h*mask
        h = self.drop(self.c2(h)); h = h*mask
        return F.gelu(h + r)

class SeqCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(4, 64, 9, padding=4)
        self.b1 = Block(64, 5, 1); self.b2 = Block(64, 5, 2); self.b3 = Block(64, 5, 4)
        self.head = nn.Sequential(nn.Linear(130, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    def forward(self, x, mask, feats):
        # x (B,4,L) float ; mask (B,1,L) float ; feats (B,2) standardized loglen/GC
        h = F.gelu(self.stem(x)); h = h*mask
        for b in (self.b1, self.b2, self.b3):
            h = b(h, mask)
        m = mask
        denom = m.sum(-1).clamp(min=1.0)
        mean = (h*m).sum(-1)/denom
        neg = torch.finfo(h.dtype).min
        mx = h.masked_fill(m <= 0, neg).max(-1).values
        z = torch.cat([mean, mx, feats], dim=1)     # 64+64+2 = 130
        return self.head(z).squeeze(-1)
