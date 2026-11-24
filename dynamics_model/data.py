"""Replay dataset D = {(s_t, a_t, s_{t+1})} for the dynamics model.

Each transition is stored ready for training, in its action frame:
  pts_in    (N_in, 3)   model input: observed cloud (block + some table points), action frame
  pts_loss  (N_loss, 3) the object's observed points (simulator segmentation), action frame;
                        only used to compute the point-flow loss
  act_feat  (3,)        (p_z, F, T)
  R_gt, t_gt            true rigid motion of the object caused by the push, action frame
                        (R_gt = I for tasks without rotation, see env.tasks)
  setting   ()          physics-setting id (indexes the latent table)
plus, for analysis only: the world-frame action, the true physics values, whether the gripper
touched the object, and a train/val flag.

Optionally (train_dynamics.py --save-images) the camera image of s_t is kept as well,
  image     (H, W, 3) uint8, world frame,
for the image-feature models. Images stay on the CPU (TransitionBuffer.images()); the frozen
DINOv2 features computed from them by scripts/dino_features.py are what the models consume
(`img_feat`, attached with attach()).
"""

from __future__ import annotations

import torch

FIELDS = ["pts_in", "pts_loss", "act_feat", "R_gt", "t_gt", "setting", "action", "physics", "made_contact", "is_val"]
OPTIONAL = ["image"]


class TransitionBuffer:
    def __init__(self, val_fraction=0.1, seed=0):
        self.data = {k: [] for k in FIELDS + OPTIONAL}
        self.val_fraction = val_fraction
        self._gen = torch.Generator().manual_seed(seed)
        self._cache = None
        self._attached = {}
        self.meta = {}

    def __len__(self):
        return sum(len(x) for x in self.data["setting"])

    def add(self, **batch):
        """Add a batch of transitions (all tensors with the same leading dim; `image` optional)."""
        n = len(batch["setting"])
        batch["is_val"] = torch.rand(n, generator=self._gen) < self.val_fraction
        for k in FIELDS:
            self.data[k].append(batch[k].detach().cpu())
        for k in OPTIONAL:
            if batch.get(k) is not None:
                self.data[k].append(batch[k].detach().cpu())
        self._cache = None

    def has(self, key):
        return bool(self.data.get(key))

    def tensors(self, device):
        """All per-transition fields concatenated, on `device` (images excluded; attached
        features included)."""
        if self._cache is None or self._cache[0] != device:
            t = {k: torch.cat(v).to(device) for k, v in self.data.items() if k in FIELDS}
            t.update({k: v.to(device) for k, v in self._attached.items()})
            self._cache = (device, t)
        return self._cache[1]

    def attach(self, key, tensor):
        """Attach a per-transition tensor (e.g. precomputed image features, (N, ...)) so that
        tensors(device)[key] returns it."""
        assert len(tensor) == len(self), f"{key}: {len(tensor)} rows for {len(self)} transitions"
        self._attached[key] = tensor
        self._cache = None

    def images(self):
        """(N, H, W, 3) uint8 on the CPU, or None if no images were stored."""
        return torch.cat(self.data["image"]) if self.has("image") else None

    def split_indices(self, device):
        t = self.tensors(device)
        idx = torch.arange(len(t["setting"]), device=device)
        return idx[~t["is_val"]], idx[t["is_val"]]

    def save(self, path, meta=None):
        d = {k: torch.cat(v) if v else None for k, v in self.data.items()}
        d["meta"] = {**self.meta, **(meta or {})}
        torch.save(d, path)

    @classmethod
    def load(cls, path, **kw):
        buf = cls(**kw)
        d = torch.load(path, weights_only=False)
        for k in FIELDS + OPTIONAL:
            buf.data[k] = [d[k]] if d.get(k) is not None else []
        buf.meta = d.get("meta", {})
        return buf

    def settings_seen(self):
        return torch.cat(self.data["setting"]).unique() if self.data["setting"] else torch.empty(0, dtype=torch.long)
