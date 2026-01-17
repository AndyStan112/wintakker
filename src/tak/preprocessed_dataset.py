import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    def __init__(self, preproc_dir):
        self.preproc_dir = preproc_dir

        with open(os.path.join(preproc_dir, "meta.json"), "r") as f:
            self.meta = json.load(f)

        self.shards = self.meta["shards"]
        self.dtype = np.float32 if self.meta["dtype"] == "float32" else np.float16

        self._current_shard_idx = None
        self.X_map = None
        self.Y_map = None
        self.G_map = None
        self.GI_map = None

        self.cum = [0]
        for s in self.shards:
            self.cum.append(self.cum[-1] + int(s["size"]))
        self.total = int(self.meta["total_samples"])

    def __len__(self):
        return self.total

    def _resolve(self, name):
        return os.path.join(self.preproc_dir, os.path.basename(name))

    def _load_shard(self, shard_idx):
        if self._current_shard_idx == shard_idx:
            return

        s = self.shards[shard_idx]
        size = int(s["size"])

        self.X_map = np.memmap(
            self._resolve(s["X"]),
            dtype=self.dtype,
            mode="r",
            shape=(size, 10, 8, 8),
        )

        self.Y_map = np.memmap(
            self._resolve(s["Y"]),
            dtype=np.float32,
            mode="r",
            shape=(size,),
        )

        self.G_map = np.memmap(
            self._resolve(s["G"]),
            dtype=np.int32,
            mode="r",
            shape=(size,),
        )

        self.GI_map = np.memmap(
            self._resolve(s["GI"]),
            dtype=np.int16,
            mode="r",
            shape=(size,),
        )

        self._current_shard_idx = shard_idx

    def _loc(self, idx):
        for si in range(len(self.shards)):
            if idx < self.cum[si + 1]:
                return si, idx - self.cum[si]
        raise IndexError(idx)

    def __getitem__(self, idx):
        si, local = self._loc(int(idx))
        self._load_shard(si)

        # numpy memmap slices
        x_np = self.X_map[local]     # shape: (10, 8, 8)
        y_val = float(self.Y_map[local])
        gid = int(self.G_map[local])
        gidx = int(self.GI_map[local])

        # Convert safely to torch tensor (2.5 KB copy, super fast)
        xt = torch.tensor(x_np, dtype=torch.float32)

        # Keep y as (1,) so collate produces (B,1)
        yt = torch.tensor([y_val], dtype=torch.float32)

        return (
            xt,
            yt,
            torch.tensor(gid, dtype=torch.int32),
            torch.tensor(gidx, dtype=torch.int32),
        )
