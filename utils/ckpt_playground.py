#!/usr/bin/env python3
import sys

sys.path.append("../src")
import torch  # noqa
from glob import glob  # noqa
import numpy as np  # noqa
from tqdm import tqdm  # noqa
from variations.render_helpers import render_rays  # noqa
from variations.nrgbd import Decoder  # noqa
from dataset.replica import DataLoader  # noqa
from frame import RGBDFrame  # noqa


torch.classes.load_library(glob("../third_party/sparse_octree/build/lib*/*.so")[0])

ckpt = None


def render_debug_images(decoder, map_states, current_frame):
    rotation = current_frame.get_rotation()
    w, h = current_frame.rgb.shape[:2]
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
        current_frame.color_embed,
        map_states,
        decoder,
        0.1,
        0.2,
        0.1,
        10,
        10,
        chunk_size=20000,
        return_raw=True,
    )

    rdepth = final_outputs["depth"]
    rcolor = final_outputs["color"]
    return rdepth.detach().cpu(), rcolor.detach().cpu()


ckpt_path = "../../archive/robust/room0_global/ckpt/final_ckpt.pth"
color_emb_path = "../../archive/robust/room0_global/misc/color_embeddings.npy"
dataset_path = "../Datasets/Replica_global/room0/"
if ckpt is None:
    ckpt = torch.load(ckpt_path)
color_emb = np.load(color_emb_path)
color_emb = torch.from_numpy(color_emb).float()
dataset = DataLoader("../Datasets/Replica_global/room0", use_gt=True)

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

for i in tqdm(range(dataset.num_imgs)):
    current_frame = RGBDFrame(*(dataset[i]))
    depth, color = render_debug_images(decoder, map_state, current_frame)
    break
