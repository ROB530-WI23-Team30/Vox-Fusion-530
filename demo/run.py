import os
import sys
import random
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("src"))  # noqa
from parser import get_parser  # noqa
from voxslam import VoxSLAM  # noqa


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    args = get_parser().parse_args()
    if hasattr(args, "seeding"):
        setup_seed(args.seeding)

    slam = VoxSLAM(args)
    slam.start()
    slam.wait_child_processes()
