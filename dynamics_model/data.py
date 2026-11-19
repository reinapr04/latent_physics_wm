"""Replay dataset D = {(s_t, a_t, s_{t+1})} for the dynamics model.

Each transition is stored ready for training, in its action frame:
  pts_in    (N_in, 3)   model input: observed cloud (table cropped away), action frame
  pts_loss  (N_loss, 3) the block's observed points (simulator segmentation), action frame;
                        only used to compute the point-flow loss
  act_feat  (3,)        (p_z, F, T)
  R_gt, t_gt            true rigid motion of the block caused by the push, action frame
  setting   ()          physics-setting id (indexes the latent table)
plus, for analysis only: the world-frame action, the true physics values, whether the gripper
touched the block, and a train/val flag.
"""

from __future__ import annotations

import torch

FIELDS = ["pts_in", "pts_loss", "act_feat", "R_gt", "t_gt", "setting", "action", "physics", "made_contact", "is_val"]


class TransitionBuffer:
    def __init__(self, val_fraction=0.1, seed=0):
        self.data = {k: [] for k in FIELDS}
        self.val_fraction = val_fraction
        self._gen = torch.Generator().manual_seed(seed)
        self._cache = None

    def __len__(self):
        return sum(len(x) for x in self.data["setting"])

    def add(self, **batch):
        """Add a batch of transitions (all tensors with the same leading dim)."""
        n = len(batch["setting"])
        batch["is_val"] = torch.rand(n, generator=self._gen) < self.val_fraction
        for k in FIELDS:
            self.data[k].append(batch[k].detach().cpu())
        self._cache = None

    def tensors(self, device):
        if self._cache is None or self._cache[0] != device:
            self._cache = (device, {k: torch.cat(v).to(device) for k, v in self.data.items()})
        return self._cache[1]

    def split_indices(self, device):
        t = self.tensors(device)
        idx = torch.arange(len(t["setting"]), device=device)
        return idx[~t["is_val"]], idx[t["is_val"]]

    def save(self, path):
        torch.save({k: torch.cat(v) if v else None for k, v in self.data.items()}, path)

    @classmethod
    def load(cls, path, **kw):
        buf = cls(**kw)
        d = torch.load(path)
        for k in FIELDS:
            buf.data[k] = [d[k]] if d[k] is not None else []
        return buf

    def settings_seen(self):
        return torch.cat(self.data["setting"]).unique() if self.data["setting"] else torch.empty(0, dtype=torch.long)
