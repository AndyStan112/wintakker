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

        self.cum = [0]
        for s in self.shards:
            self.cum.append(self.cum[-1] + int(s["size"]))
        self.total = int(self.meta["total_samples"])

        self._current_shard_idx = None

        self.X_tensor = None
        self.Y_tensor = None
        self.G_tensor = None
        self.GI_tensor = None

    def __len__(self):
        return self.total

    def _resolve(self, name):
        return os.path.join(self.preproc_dir, os.path.basename(name))

    def _loc(self, idx):
        for si in range(len(self.shards)):
            if idx < self.cum[si + 1]:
                return si, idx - self.cum[si]
        raise IndexError(idx)

    def _load_shard(self, shard_idx):
        if self._current_shard_idx == shard_idx:
            return

        s = self.shards[shard_idx]
        size = int(s["size"])

        print(f"[dataset] loading shard {shard_idx} into RAM ({size} samples)")

        X_mm = np.memmap(
            self._resolve(s["X"]),
            dtype=np.float32,
            mode="r",
            shape=(size, 10, 8, 8),
        )
        Y_mm = np.memmap(
            self._resolve(s["Y"]),
            dtype=np.float32,
            mode="r",
            shape=(size,),
        )
        G_mm = np.memmap(
            self._resolve(s["G"]),
            dtype=np.int32,
            mode="r",
            shape=(size,),
        )
        GI_mm = np.memmap(
            self._resolve(s["GI"]),
            dtype=np.int16,
            mode="r",
            shape=(size,),
        )

        X_np = np.array(X_mm, dtype=np.float32)
        Y_np = np.array(Y_mm, dtype=np.float32)
        G_np = np.array(G_mm, dtype=np.int32)
        GI_np = np.array(GI_mm, dtype=np.int16)


        self.X_tensor = torch.from_numpy(X_np)
        self.Y_tensor = torch.from_numpy(Y_np).unsqueeze(1)
        self.G_tensor = torch.from_numpy(G_np)
        self.GI_tensor = torch.from_numpy(GI_np)

        self._current_shard_idx = shard_idx

    def __getitem__(self, idx):
        idx = int(idx)
        shard_idx, local = self._loc(idx)

        if self._current_shard_idx != shard_idx:
            self._load_shard(shard_idx)

        return (
            self.X_tensor[local],
            self.Y_tensor[local],
            self.G_tensor[local],
            self.GI_tensor[local],
        )
