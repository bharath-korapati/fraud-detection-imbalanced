"""Unsupervised detectors: learn what *normal* looks like, flag what doesn't fit.

Neither model is shown a single fraud label while training. That's the point:
in production, brand-new fraud patterns have no labels yet.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch import nn

SEED = 42


def isolation_forest_scores(X_train, y_train, X_eval) -> np.ndarray:
    """Higher = weirder. Trained on legit rows only."""
    normal = np.asarray(X_train)[np.asarray(y_train) == 0]
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED, n_jobs=-1)
    iso.fit(normal)
    return -iso.score_samples(np.asarray(X_eval))


class _AE(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def autoencoder_scores(X_train, y_train, X_eval, epochs: int = 15, batch: int = 512) -> np.ndarray:
    """Squeeze each legit transaction through an 8-number bottleneck and rebuild it.

    Score = reconstruction error. A transaction unlike anything in training
    rebuilds badly -> high score.
    """
    torch.manual_seed(SEED)
    normal = np.asarray(X_train, dtype=np.float32)[np.asarray(y_train) == 0]
    scaler = StandardScaler().fit(normal)
    Xn = torch.tensor(scaler.transform(normal), dtype=torch.float32)
    Xe = torch.tensor(scaler.transform(np.asarray(X_eval, dtype=np.float32)), dtype=torch.float32)

    model = _AE(Xn.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        perm = torch.randperm(len(Xn), generator=gen)
        for i in range(0, len(Xn), batch):
            xb = Xn[perm[i:i + batch]]
            opt.zero_grad()
            loss_fn(model(xb), xb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return ((model(Xe) - Xe) ** 2).mean(dim=1).numpy()
