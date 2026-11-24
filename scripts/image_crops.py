"""Action-frame image crops: the image analogue of the action-frame point cloud.

The camera is fixed and known, so every pixel of the table plane can be mapped to a world
position. For each stored transition this samples the camera image on a grid laid out in the
push's action frame (origin at the contact point, x along the push), on the plane at the
object's center height, i.e. a top-down view of the table rotated and centred on the push.
The object is then always seen the same way relative to the push, whatever the push's world
position and direction - the equivariance the point cloud gets for free. (The view is exact
for the plane; the object's sides lean away from the camera, so the crop still carries the
world direction of the push, which the model also receives.)

Writes <dataset dir>/image_af.pt with
  af224    (N, 224, 224, 3) uint8  action-frame crop, CROP_X x CROP_Y m, for frozen DINOv2
                                    (scripts/dino_features.py --images ... --key af224)
  af96     (N, 96, 96, 3)   uint8  the same at 96 px, for a CNN trained from scratch (image_af_cnn)
  world96  (N, 96, 96, 3)   uint8  the world-frame image at 96 px (image_cnn)
plus a preview png of a few crops. Row index <-> -y (row 0 is the +y side), so mirroring a
push (y -> -y) is a vertical flip of the crop.

    python scripts/image_crops.py --dataset runs/block/dataset.pt
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from dynamics_model.data import TransitionBuffer  # noqa: E402

from dynamics_model.inputs import CROP_X, CROP_Y, af_crops as crops, resize96  # noqa: E402


def camera_from_env(task, path):
    """Create one env of the task and save its camera intrinsics (OpenCV convention), extrinsics
    (world -> camera, 3 x 4) and image size to `path`."""
    from env import make_push_env
    from env.observation import OBS_MODE
    from dynamics_model.inputs import camera_from_obs

    env = make_push_env(num_envs=1, obs_mode=OBS_MODE, task=task)
    obs, _ = env.reset(seed=0)
    cam = camera_from_obs(obs)
    env.close()
    Path(path).write_text(json.dumps(cam, indent=2))
    return cam


def preview(images, af, actions, path, n=6):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    for i in range(n):
        axes[0, i].imshow(images[i].numpy())
        axes[0, i].set_title(f"world  d=({actions[i, 3]:+.1f},{actions[i, 4]:+.1f})", fontsize=7)
        axes[1, i].imshow(af[i].numpy())
        axes[1, i].set_title("action frame (push -> right)", fontsize=7)
        for ax in axes[:, i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--camera", default="runs/camera.json", help="camera file (created from a live env if missing)")
    ap.add_argument("--out", default=None, help="default: <dataset dir>/image_af.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    buf = TransitionBuffer.load(args.dataset)
    task = buf.meta.get("task", "block")
    from env import get_task

    plane_z = get_task(task).half_size
    cam = json.loads(Path(args.camera).read_text()) if Path(args.camera).exists() else camera_from_env(task, args.camera)
    images = buf.images()
    assert images is not None, "this dataset has no images (collect with --save-images)"
    actions = torch.cat(buf.data["action"])
    t0 = time.time()
    af224, af96, w96 = [], [], []
    for i in range(0, len(images), 256):
        img = images[i:i + 256].to(args.device)
        a = actions[i:i + 256].to(args.device)
        af224.append(crops(img, a, 224, plane_z, cam).cpu())
        af96.append(crops(img, a, 96, plane_z, cam).cpu())
        w96.append(resize96(img).cpu())
    d = {"af224": torch.cat(af224), "af96": torch.cat(af96), "world96": torch.cat(w96),
         "crop_x": CROP_X, "crop_y": CROP_Y, "plane_z": plane_z, "camera": cam}
    out = Path(args.out) if args.out else Path(args.dataset).parent / "image_af.pt"
    torch.save(d, out)
    preview(images, d["af224"], actions, out.with_suffix(".png"))
    print(f"{len(images)} images -> crops in {time.time() - t0:.0f} s, saved {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
