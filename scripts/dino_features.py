"""Frozen DINOv2 features of a dataset's camera images, for the image-feature dynamics models.

For every stored transition image (dynamics_model.data, 224 x 224 RGB) this computes the
DINOv2 ViT-S/14 tokens [CLS, 16 x 16 patches] (257 x 384, fp16) with the backbone frozen, and
saves them next to the dataset. scripts/ablate.py attaches them to the buffer as `img_feat`;
eval_mpc.py / eval_latent_inference.py call compute_tokens() online.

    python scripts/dino_features.py --dataset runs/block/dataset.pt
    python scripts/dino_features.py --dataset runs/block/dataset.pt --images runs/block/image_af.pt --key af224 \
        --out runs/block/dino_af_feats.pt          # tokens of the action-frame crops (scripts/image_crops.py)
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamics_model.data import TransitionBuffer  # noqa: E402

MODEL = "facebook/dinov2-small"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
_cache = {}


def load_backbone(device):
    if device not in _cache:
        from transformers import AutoModel

        m = AutoModel.from_pretrained(MODEL).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _cache[device] = m
    return _cache[device]


@torch.no_grad()
def compute_tokens(images, device, batch=128):
    """images (N, H, W, 3) uint8 -> (N, 257, 384) fp16 tokens on `device`."""
    m = load_backbone(device)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), batch):
        x = images[i:i + batch].to(device).permute(0, 3, 1, 2).float() / 255
        if x.shape[-1] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", antialias=True)
        x = (x - mean) / std
        out.append(m(pixel_values=x).last_hidden_state.half())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None, help="default: <dataset dir>/dino_feats.pt")
    ap.add_argument("--images", default=None, help="a .pt file of images to encode instead of the dataset's (image_crops.py)")
    ap.add_argument("--key", default="af224", help="key of the images in --images")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    buf = TransitionBuffer.load(args.dataset)
    if args.images:
        imgs = torch.load(args.images)[args.key]
        assert args.out, "--out is required with --images"
    else:
        imgs = buf.images()
    assert imgs is not None, "this dataset has no images (collect with --save-images)"
    t0 = time.time()
    tok = compute_tokens(imgs, args.device)
    out = Path(args.out) if args.out else Path(args.dataset).parent / "dino_feats.pt"
    torch.save({"tokens": tok.cpu(), "model": MODEL, "n": len(buf)}, out)
    print(f"{len(imgs)} images -> tokens {tuple(tok.shape)} in {time.time() - t0:.0f} s, saved {out}")


if __name__ == "__main__":
    main()
