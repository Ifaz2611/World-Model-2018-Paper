import glob

import numpy as np
import torch
from torch.utils.data import Dataset

ROLLOUT_DIR = "data/rollouts"


def load_rollouts(rollout_dir=ROLLOUT_DIR, limit=None):
    paths = sorted(glob.glob(f"{rollout_dir}/rollout_*.npz"))

    if (limit is not None):
        paths = paths[:limit]

    return paths


class RolloutFrames(Dataset):
    def __init__(self, rollout_dir=ROLLOUT_DIR, limit=None):
        paths = load_rollouts(rollout_dir, limit)

        # The V model is trained on frames with no notion of which rollout or what order
        # they came from, so everything collapses into one stack up front. 50k 64x64
        # uint8 frames is under a gigabyte, so it all stays resident.
        self.frames = np.concatenate([np.load(path)["frames"] for path in paths])

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        # frame shape: (64, 64, 3) uint8 -> (3, 64, 64) float in [0, 1]
        frame = torch.from_numpy(self.frames[idx]).permute(2, 0, 1)

        return frame.float() / 255.0
