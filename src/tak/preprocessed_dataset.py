import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """Map-style dataset backed by sharded numpy memmaps produced by the
    preprocessing script.

    meta.json format:
      {
        'total_samples': N,
        'dtype': 'float16'|'float32',
        'shape': [10,8,8],
        'shards': [ { 'X': 'X_0.dat', 'Y': 'Y_0.dat', 'G': ..., 'GI': ..., 'size': n0 }, ... ]
      }
    """

    def __init__(self, preproc_dir):
        meta_path = os.path.join(preproc_dir, "meta.json")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.shards = self.meta["shards"]

        self.X_maps = []
        self.Y_maps = []
        self.G_maps = []
        self.GI_maps = []
        for s in self.shards:
            self.X_maps.append(
                np.memmap(
                    s["X"],
                    dtype=np.float16 if self.meta["dtype"] == "float16" else np.float32,
                    mode="r",
                    shape=(s["size"], 10, 8, 8),
                )
            )
            self.Y_maps.append(
                np.memmap(s["Y"], dtype=np.float32, mode="r", shape=(s["size"],))
            )
            self.G_maps.append(
                np.memmap(s["G"], dtype=np.int32, mode="r", shape=(s["size"],))
            )
            self.GI_maps.append(
                np.memmap(s["GI"], dtype=np.int16, mode="r", shape=(s["size"],))
            )

        self.cum = [0]
        for s in self.shards:
            self.cum.append(self.cum[-1] + int(s["size"]))
        self.total = int(self.meta["total_samples"])

    def __len__(self):
        return self.total

    def _loc(self, idx):

        for si in range(len(self.shards)):
            if idx < self.cum[si + 1]:
                local = idx - self.cum[si]
                return si, int(local)
        raise IndexError(idx)

    def __getitem__(self, idx):
        si, local = self._loc(int(idx))
        x = self.X_maps[si][local]
        y = self.Y_maps[si][local]
        gid = int(self.G_maps[si][local])
        gidx = int(self.GI_maps[si][local])
        xt = torch.from_numpy(np.array(x)).to(dtype=torch.float32)
        yt = torch.tensor([float(y)], dtype=torch.float32)
        return (
            xt,
            yt,
            torch.tensor(gid, dtype=torch.int32),
            torch.tensor(gidx, dtype=torch.int32),
        )
