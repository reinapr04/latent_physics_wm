"""Model inputs other than the action-frame point cloud, computed online from the camera image:
world-frame DINOv2 tokens, action-frame crops (scripts/image_crops.py explains them), their
tokens, and 96 px images for the CNNs. Used by the evaluation scripts (eval_latent_inference.py,
eval_mpc.py) for the image-based models; the offline ablations precompute the same things.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dynamics_model.geometry import action_frame

CROP_X = (-0.08, 0.22)  # m, action frame: from 8 cm behind the contact point to 22 cm ahead
CROP_Y = (-0.15, 0.15)


def camera_from_obs(obs):
    """Camera intrinsics / extrinsics (OpenCV convention) and resolution from a live observation."""
    sp = obs["sensor_param"]["base_camera"]
    rgb = obs["sensor_data"]["base_camera"]["rgb"] if "sensor_data" in obs else None
    cam = {"K": sp["intrinsic_cv"][0].cpu().tolist(), "E": sp["extrinsic_cv"][0].cpu().tolist()}
    if rgb is not None:
        cam["height"], cam["width"] = int(rgb.shape[1]), int(rgb.shape[2])
    else:
        cam["height"] = cam["width"] = int(round(2 * cam["K"][0][2]))
    return cam


def action_frame_grid(actions, size, plane_z, cam, img_hw):
    """Sampling grid (B, size, size, 2) in normalized image coordinates (F.grid_sample,
    align_corners=False) of a size x size action-frame window on the plane z = plane_z.
    Row 0 is the +y side, columns run along the push (+x to the right)."""
    dev = actions.device
    B = len(actions)
    xs = torch.linspace(CROP_X[0], CROP_X[1], size, device=dev)
    ys = torch.linspace(CROP_Y[1], CROP_Y[0], size, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx, gy, torch.full_like(gx, plane_z)], -1).reshape(1, -1, 3)
    origin, R_a = action_frame(actions)
    world = pts @ R_a.transpose(1, 2) + origin[:, None]  # (B, S*S, 3)
    K = torch.tensor(cam["K"], device=dev, dtype=torch.float32)
    E = torch.tensor(cam["E"], device=dev, dtype=torch.float32)
    c = world @ E[:, :3].T + E[:, 3]
    uv = c @ K.T
    u, v = uv[..., 0] / uv[..., 2].clamp_min(1e-6), uv[..., 1] / uv[..., 2].clamp_min(1e-6)
    h, w = img_hw
    u = u * w / cam["width"]  # the stored image may be resized
    v = v * h / cam["height"]
    return torch.stack([2 * u / w - 1, 2 * v / h - 1], -1).reshape(B, size, size, 2)


def af_crops(images, actions, size, plane_z, cam):
    """images (B, H, W, 3) uint8, actions (B, 7) -> (B, size, size, 3) uint8 action-frame crops."""
    grid = action_frame_grid(actions, size, plane_z, cam, images.shape[1:3])
    x = images.permute(0, 3, 1, 2).float()
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def resize96(images):
    x = F.interpolate(images.permute(0, 3, 1, 2).float(), size=(96, 96), mode="area")
    return x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def model_inputs(model, images, actions, cam, half_size, dev, chunk=256):
    """The input `x` (and aux) of an image-based model for images (N, H, W, 3) uint8 (CPU or
    device) and actions (N, 7): frozen DINOv2 tokens, crops, or 96 px images as the encoder
    needs (dynamics_model.model.ENCODER_IO). Returns (x, aux)."""
    key, aux_key, _ = model.input_key, model.aux_key, model.aug_kind
    out = {}
    for k in [key] + ([aux_key] if aux_key else []):
        if k == "pts_in":
            continue
        parts = []
        for i in range(0, len(images), chunk):
            img = images[i:i + chunk].to(dev)
            a = actions[i:i + chunk].to(dev)
            if k == "img_feat":
                from scripts.dino_features import compute_tokens
                parts.append(compute_tokens(img, dev))
            elif k == "img_af_feat":
                from scripts.dino_features import compute_tokens
                parts.append(compute_tokens(af_crops(img, a, 224, half_size, cam), dev))
            elif k == "image_96":
                parts.append(resize96(img))
            elif k == "image_af_96":
                parts.append(af_crops(img, a, 96, half_size, cam))
            else:
                raise KeyError(k)
        out[k] = torch.cat(parts)
    return out.get(key), out.get(aux_key) if aux_key else None


def attach_inputs(buf, model, cam, half_size, dev):
    """Attach to a TransitionBuffer the inputs an image-based model needs for its transitions."""
    if model.input_kind == "points" and not model.aux_key:
        return
    images = buf.images()
    assert images is not None, "the buffer has no images (collect with save_images)"
    actions = torch.cat(buf.data["action"])
    x, aux = model_inputs(model, images, actions, cam, half_size, dev)
    if x is not None:
        buf.attach(model.input_key, x)
    if aux is not None:
        buf.attach(model.aux_key, aux)


def planner_inputs(model, images, cam, half_size, dev):
    """inputs_fn(env_idx (M,), actions (M, 7)) -> (x, aux) for policy.mpc.random_shooting_mpc:
    the model's input for candidate pushes `actions` in the environments `env_idx`, from the
    current camera images (B, H, W, 3) of all environments. World-frame inputs are computed
    once per environment; action-frame crops depend on the candidate and are computed per call."""
    if model.input_kind == "points" and not model.aux_key:
        return None
    B = len(images)
    images = images.to(dev)
    fixed = {}
    for k in [model.input_key] + ([model.aux_key] if model.aux_key else []):
        if k in ("img_feat", "image_96"):
            fixed[k] = model_inputs_key(k, images, None, cam, half_size, dev)

    def per_action(k, env_idx, actions):
        if k in fixed:
            return fixed[k][env_idx]
        if k == "pts_in":
            return None
        return model_inputs_key(k, images[env_idx], actions, cam, half_size, dev)

    def fn(env_idx, actions):
        x = per_action(model.input_key, env_idx, actions)
        aux = per_action(model.aux_key, env_idx, actions) if model.aux_key else None
        return x, aux

    return fn


def model_inputs_key(k, images, actions, cam, half_size, dev):
    """One input kind for images (M, H, W, 3) on `dev` (actions (M, 7) for action-frame kinds)."""
    if k == "img_feat":
        from scripts.dino_features import compute_tokens
        return compute_tokens(images, dev)
    if k == "img_af_feat":
        from scripts.dino_features import compute_tokens
        return compute_tokens(af_crops(images, actions, 224, half_size, cam), dev)
    if k == "image_96":
        return resize96(images)
    if k == "image_af_96":
        return af_crops(images, actions, 96, half_size, cam)
    raise KeyError(k)
