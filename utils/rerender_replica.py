#!/usr/bin/env python3
"""
rerender results for replica dataset

Usage:
python rerender_replica.py /path/to/log_dir /path/to/dataset
"""
import sys

sys.path.append("../src")
import torch  # noqa
from glob import glob  # noqa
import numpy as np  # noqa
from tqdm import tqdm  # noqa
from variations.render_helpers import render_rays  # noqa
from variations.nrgbd import Decoder  # noqa
from variations.render_helpers import fill_in  # noqa
from dataset.replica import DataLoader  # noqa
from frame import RGBDFrame  # noqa
import cv2  # noqa
import os.path as osp  # noqa
import os  # noqa


torch.classes.load_library(glob("../third_party/sparse_octree/build/lib*/*.so")[0])

# Settings
save_dir = "./output"


def render_debug_images(decoder, map_states, current_frame, color_emb=None):
    rotation = current_frame.get_rotation()
    w, h = 640, 480
    final_outputs = dict()

    for k, v in map_states.items():
        map_states[k] = v.cuda()

    rays_d = current_frame.get_rays(w, h).cuda()
    rays_d = rays_d @ rotation.transpose(-1, -2).cuda()

    rays_o = current_frame.get_translation().cuda()
    rays_o = rays_o.unsqueeze(0).expand_as(rays_d)

    rays_o = rays_o.reshape(1, -1, 3).contiguous()
    rays_d = rays_d.reshape(1, -1, 3)

    final_outputs = render_rays(
        rays_o,
        rays_d,
        color_emb,
        map_states,
        decoder,
        0.1 * 0.2,
        0.2,
        0.1,
        10,
        10,
        chunk_size=20000,
        return_raw=True,
    )

    rdepth = final_outputs["depth"]
    rcolor = final_outputs["color"]

    rdepth = fill_in(
        (h, w, 1), final_outputs["ray_mask"].view(h, w), final_outputs["depth"], 0
    )
    rcolor = fill_in(
        (h, w, 3), final_outputs["ray_mask"].view(h, w), final_outputs["color"], 0
    )
    return rdepth.detach().cpu(), rcolor.detach().cpu()


def log_raw_image(ind, rgb, depth):
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    rgb = cv2.cvtColor(rgb * 255, cv2.COLOR_RGB2BGR)
    save_dir = "./output"
    cv2.imwrite(
        osp.join(save_dir, "frame_{:05d}.jpg".format(ind)), (rgb).astype(np.uint8)
    )
    cv2.imwrite(
        osp.join(save_dir, "depth_{:05d}.png".format(ind)),
        (depth * 5000).astype(np.uint16),
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rerender_replica.py /path/to/log_dir /path/to/dataset")
        exit()

    log_dir = sys.argv[1]
    dataset_path = sys.argv[2]
    ckpt_path = osp.join(log_dir, "ckpt/final_ckpt.pth")
    color_emb_path = osp.join(log_dir, "misc/color_embeddings.npy")
    ckpt = torch.load(ckpt_path)
    color_emb = np.load(color_emb_path)
    color_emb = torch.from_numpy(color_emb).float().cuda()
    dataset = DataLoader(dataset_path, use_gt=True)

    # load decoder
    decoder = Decoder(
        depth=2,
        width=128,
        in_dim=16,
        skips=[],
        embedder="none",
        multires=0,
        affine_color_dim=10,
    )
    decoder.load_state_dict(ckpt["decoder_state"])
    decoder = decoder.cuda()
    svo = ckpt["svo"]
    map_state = ckpt["map_state"]

    try:
        os.makedirs(save_dir)
    except:  # noqa
        pass

    for i in tqdm(range(dataset.num_imgs)):
        current_frame = RGBDFrame(*(dataset[i]))
        depth, color = render_debug_images(
            decoder, map_state, current_frame, color_emb=color_emb[0, :]
        )
        log_raw_image(i, color, depth)
