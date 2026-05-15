#src/pipeline.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from geometry import Geometry
from rays_data import RaysData
from volume_rendering import VolumeRender
from network import NeRF
from utils.utils_math import PSNR


def resize_rgb_batch(images_uint8_rgb, scale):
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if scale == 1.0:
        return images_uint8_rgb

    out = []
    for img in images_uint8_rgb:
        h, w = img.shape[:2]
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        out.append(cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA))
    return np.stack(out, axis=0)


def train(model, rays_data, near, far, n_iters, n_rays, n_samples, lr, lr_decay, perturb, grad_clip, log_every, save_every, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    renderer = VolumeRender((far - near) / n_samples)
    mse = nn.MSELoss()

    def lr_lambda(step):
        return 0.1 ** (step / (lr_decay * 1000))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    losses, psnrs = [], []

    for iteration in tqdm(range(1, n_iters + 1), desc="training"):
        ray_o, ray_d, gt = rays_data.sample_rays(n_rays)
        ray_o = torch.from_numpy(ray_o).to(device)
        ray_d = torch.from_numpy(ray_d).to(device)
        gt = torch.from_numpy(gt).to(device)

        pts_np, dirs_np, _ = rays_data.sample_along_rays(
            ray_o.detach().cpu().numpy(),
            ray_d.detach().cpu().numpy(),
            near,
            far,
            n_samples,
            perturbation=perturb,
            return_directions=True,
            return_t=True,
        )

        pts = torch.from_numpy(pts_np).to(device)
        dirs = torch.from_numpy(dirs_np).to(device)

        B, S, _ = pts.shape
        pts_flat = pts.reshape(B * S, 3)
        dirs_flat = dirs.reshape(B * S, 3)
        dirs_flat = dirs_flat / dirs_flat.norm(dim=-1, keepdim=True)

        rgb_flat, sigma_flat = model(pts_flat, dirs_flat)
        rgb = rgb_flat.view(B, S, 3)
        sigma = sigma_flat.view(B, S, 1)

        pred, _ = renderer.render(sigma, rgb)
        loss = mse(pred, gt)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        losses.append(float(loss.item()))

        if iteration % log_every == 0:
            p = float(PSNR(loss).item())
            psnrs.append(p)
            tqdm.write(
                f"iter {iteration:6d} | loss {loss.item():.6f} | PSNR {p:.2f} dB | lr {scheduler.get_last_lr()[0]:.2e}"
            )

        if iteration % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"model_{iteration:07d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            tqdm.write(f"saved {ckpt_path}")

    return model, losses, psnrs


def render_image(model, pose, intrinsics_vec, image_size_vec, near, far, n_samples=64, chunk=4096, device="cuda"):
    W, H = int(image_size_vec[0]), int(image_size_vec[1])
    fx, fy, cx, cy = intrinsics_vec
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    c2w = np.vstack([pose, [0, 0, 0, 1]]).astype(np.float32) if pose.shape == (3, 4) else pose.astype(np.float32)

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    uv = np.stack([xx + 0.5, yy + 0.5], axis=-1).reshape(-1, 2)

    ray_o_single, ray_d = Geometry.pixel_to_ray(K, c2w, uv)
    ray_o = np.broadcast_to(ray_o_single[None], ray_d.shape).copy()

    t_vals = np.linspace(near, far, n_samples, dtype=np.float32)
    renderer = VolumeRender((far - near) / n_samples)

    model.eval()
    rendered_chunks = []

    with torch.no_grad():
        for i in range(0, H * W, chunk):
            o = torch.tensor(ray_o[i : i + chunk], dtype=torch.float32, device=device)
            d = torch.tensor(ray_d[i : i + chunk], dtype=torch.float32, device=device)
            t = torch.tensor(t_vals, dtype=torch.float32, device=device)

            B = o.shape[0]
            pts = o[:, None, :] + d[:, None, :] * t[None, :, None]
            dirs = d[:, None, :].expand_as(pts)

            pts_flat = pts.reshape(B * n_samples, 3)
            dirs_flat = dirs.reshape(B * n_samples, 3)
            dirs_flat = dirs_flat / dirs_flat.norm(dim=-1, keepdim=True)

            rgb_flat, sigma_flat = model(pts_flat, dirs_flat)
            rgb = rgb_flat.view(B, n_samples, 3)
            sigma = sigma_flat.view(B, n_samples, 1)

            color, _ = renderer.render(sigma, rgb)
            rendered_chunks.append(color.cpu())

    return np.clip(torch.cat(rendered_chunks, dim=0).view(H, W, 3).numpy(), 0.0, 1.0)


def load_dataset(npz_path, training_extra_scale=1.0):
    data = np.load(npz_path)
    images_train = data["images_train"]
    c2ws_train = data["c2ws_train"].astype(np.float32)

    images_val = data["images_val"] if "images_val" in data else None
    c2ws_val = data["c2ws_val"].astype(np.float32) if "c2ws_val" in data else None

    if "intrinsics" not in data:
        raise ValueError("Expected 'intrinsics' in dataset .npz. Rebuild dataset with photo_calibration/main.py")

    fx, fy, cx, cy = data["intrinsics"].astype(np.float32)

    images_train = resize_rgb_batch(images_train, training_extra_scale)
    if images_val is not None and len(images_val) > 0:
        images_val = resize_rgb_batch(images_val, training_extra_scale)

    images_train_f = images_train.astype(np.float32) / 255.0
    images_val_f = images_val.astype(np.float32) / 255.0 if images_val is not None and len(images_val) > 0 else None

    fx *= training_extra_scale
    fy *= training_extra_scale
    cx *= training_extra_scale
    cy *= training_extra_scale

    H, W = images_train_f.shape[1], images_train_f.shape[2]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    return {
        "images_train": images_train_f,
        "c2ws_train": c2ws_train,
        "images_val": images_val_f,
        "c2ws_val": c2ws_val,
        "intrinsics": np.array([fx, fy, cx, cy], dtype=np.float32),
        "image_size": np.array([W, H], dtype=np.int32),
        "K": K,
    }


def build_default_model():
    return NeRF(pos_L=10, dir_L=4, width=256)


def build_rays_data(images_train, K, c2ws_train):
    return RaysData(images_train, K, c2ws_train)
