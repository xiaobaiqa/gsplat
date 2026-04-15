# SPDX-License-Identifier: Apache-2.0
"""Render an RGB-only trajectory video from a gsplat checkpoint."""

import argparse
import math
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from datasets.colmap import Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from gsplat.rendering import rasterization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to ckpt_*.pt")
    parser.add_argument("--data_dir", required=True, help="COLMAP/MipNeRF360 scene dir")
    parser.add_argument("--output", required=True, help="Output .mp4 path")
    parser.add_argument("--duration", type=float, default=15.0, help="Video seconds")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--traj", choices=["ellipse", "interp", "spiral"], default="ellipse")
    parser.add_argument("--sh_degree", type=int, default=None)
    parser.add_argument("--near_plane", type=float, default=0.01)
    parser.add_argument("--far_plane", type=float, default=1e10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scene_parser = Parser(
        data_dir=args.data_dir,
        factor=args.factor,
        normalize=True,
        test_every=8,
        load_exposure=False,
    )
    camtoworlds = scene_parser.camtoworlds[5:-5]
    num_frames = max(2, int(round(args.duration * args.fps)))

    if args.traj == "interp":
        n_interp = max(1, math.ceil(num_frames / max(len(camtoworlds) - 1, 1)))
        camtoworlds = generate_interpolated_path(camtoworlds, n_interp)
        src = np.linspace(0, len(camtoworlds) - 1, num_frames)
        camtoworlds = camtoworlds[np.round(src).astype(int)]
    elif args.traj == "ellipse":
        height = camtoworlds[:, 2, 3].mean()
        camtoworlds = generate_ellipse_path_z(
            camtoworlds,
            n_frames=num_frames,
            height=height,
        )
    elif args.traj == "spiral":
        camtoworlds = generate_spiral_path(
            camtoworlds,
            bounds=scene_parser.bounds * scene_parser.scene_scale,
            n_frames=num_frames,
            spiral_scale_r=scene_parser.extconf["spiral_radius_scale"],
        )

    bottom = np.repeat(
        np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0
    )
    camtoworlds = np.concatenate([camtoworlds, bottom], axis=1)
    camtoworlds = torch.from_numpy(camtoworlds).float().to(device)

    ckpt = torch.load(args.ckpt, map_location=device)["splats"]
    means = ckpt["means"]
    quats = F.normalize(ckpt["quats"], p=2, dim=-1)
    scales = torch.exp(ckpt["scales"])
    opacities = torch.sigmoid(ckpt["opacities"])
    colors = torch.cat([ckpt["sh0"], ckpt["shN"]], dim=-2)
    sh_degree = (
        min(args.sh_degree, int(math.sqrt(colors.shape[-2]) - 1))
        if args.sh_degree is not None
        else int(math.sqrt(colors.shape[-2]) - 1)
    )

    K = torch.from_numpy(list(scene_parser.Ks_dict.values())[0]).float().to(device)
    width, height = list(scene_parser.imsize_dict.values())[0]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = imageio.get_writer(args.output, fps=args.fps)
    with torch.no_grad():
        for i in tqdm.trange(len(camtoworlds), desc="Rendering RGB trajectory"):
            renders, _, _ = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                torch.linalg.inv(camtoworlds[i : i + 1]),
                K[None],
                width,
                height,
                sh_degree=sh_degree,
                near_plane=args.near_plane,
                far_plane=args.far_plane,
                render_mode="RGB",
                packed=False,
            )
            frame = torch.clamp(renders[0, ..., :3], 0.0, 1.0).cpu().numpy()
            writer.append_data((frame * 255).astype(np.uint8))
    writer.close()
    print(f"Video saved to {args.output}")


if __name__ == "__main__":
    main()
