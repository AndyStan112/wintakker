import numpy as np
import json
import os

PREPROC = "preproc"
CHUNK = 10000


def resolve_path(base, p):
    """
    Returns a correct filepath:
    - If p is absolute → return as-is.
    - If p already contains subdirectories → join base with only the basename.
    - If p is simple filename → join base + p.
    """
    p = p.replace("/", os.sep).replace("\\", os.sep)

    if os.path.isabs(p):
        return p

    if os.path.dirname(p):
        return os.path.join(base, os.path.basename(p))

    return os.path.join(base, p)


def convert_shard(old_path, new_path, size):
    shape = (size, 10, 8, 8)

    print(f"\n[+] Converting:")
    print(f"    old = {old_path}")
    print(f"    new = {new_path}")
    print(f"    size = {size}")

    mm16 = np.memmap(old_path, dtype=np.float16, mode="r", shape=shape)
    mm32 = np.memmap(new_path, dtype=np.float32, mode="w+", shape=shape)

    chunks = (size + CHUNK - 1) // CHUNK

    for ci in range(chunks):
        start = ci * CHUNK
        end = min(start + CHUNK, size)
        print(f"    chunk {ci+1}/{chunks} ({start}:{end})")

        mm32[start:end] = mm16[start:end].astype(np.float32, copy=False)

    del mm16
    del mm32

    print("    ✓ Done.")


def main():
    meta_path = os.path.join(PREPROC, "meta.json")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    if meta["dtype"] == "float32":
        print("Already float32 — nothing to convert.")
        return

    shards = meta["shards"]

    for idx, shard in enumerate(shards):
        print(f"\n============================")
        print(f"  Shard {idx+1}/{len(shards)}")
        print(f"============================")

        old_full = resolve_path(PREPROC, shard["X"])
        size = int(shard["size"])

        new_name = os.path.basename(shard["X"]).replace(".dat", "_fp32.dat")
        new_full = os.path.join(PREPROC, new_name)

        print(f"Resolved old path: {old_full}")
        print(f"Resolved new path: {new_full}")

        if os.path.exists(new_full):
            print("FP32 output exists, skipping conversion.")
        else:
            convert_shard(old_full, new_full, size)

        # verify correct size
        expected_bytes = size * 10 * 8 * 8 * 4
        actual_bytes = os.path.getsize(new_full)

        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"ERROR: Size mismatch in {new_full}. "
                f"Expected {expected_bytes}, got {actual_bytes}"
            )

        print("    ✓ Verification passed.")

        print(f"    Deleting old FP16 file: {old_full}")
        os.remove(old_full)

        shard["X"] = new_name

    meta["dtype"] = "float32"

    print("\nUpdating meta.json...")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✓ All shards converted safely.")
    print("✓ meta.json updated.")
    print("✓ FP16 shards removed.")


if __name__ == "__main__":
    main()
