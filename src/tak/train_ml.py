import argparse
import signal
import sys
import time
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, SequentialSampler


from tak.preprocessed_dataset import PreprocessedDataset
torch.set_num_threads(1)


try:
    from tak.preprocessed_dataset import PreprocessedDataset
except Exception:
    PreprocessedDataset = None
from tak.ml import ValueNet
from tqdm import tqdm

import math
import multiprocessing


def save_checkpoint(path, net, opt, epoch, step, extra=None):
    ckpt = {
        "epoch": epoch,
        "step": step,
        "state_dict": net.state_dict(),
        "optimizer": opt.state_dict(),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path, net, opt, device):
    data = torch.load(path, map_location=device)
    net.load_state_dict(data["state_dict"])
    if opt is not None and "optimizer" in data:
        try:
            opt.load_state_dict(data["optimizer"])
        except Exception:
            print("warning: failed to load optimizer state; continuing")
    epoch = int(data.get("epoch", 0))
    step = int(data.get("step", 0))
    completed_epoch = bool(data.get("completed_epoch", False))

    return epoch, step, completed_epoch, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tak_value_net.pt")
    ap.add_argument("--checkpoint", default="tak_value_net.ckpt")
    ap.add_argument("--resume", default="")
    ap.add_argument("--restore", dest="resume", default=None, help="alias for --resume")
    ap.add_argument("--sizes", default="")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument(
        "--add-epochs",
        type=int,
        default=0,
        help="when resuming, run this many additional epochs beyond the checkpoint's epoch",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--samples-per-game", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--amp",
        action="store_true",
        help="enable automatic mixed precision (FP16) if GPU is available",
    )
    ap.add_argument(
        "--preprocessed-dir",
        default="",
        help="path to preprocessed dataset directory (meta.json + shard files)",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="limit the number of samples used for training (0 = use all); useful for quick tests",
    )
    ap.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="save checkpoint every N epochs (default 1)",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    meta_path = os.path.join(args.preprocessed_dir, "meta.json")
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        total_samples = int(meta.get("total_samples", 0))
    except Exception:
        total_samples = 0
    total_batches = max(1, math.ceil(max(1, total_samples) / args.batch))
    print(
        f"preprocessed dataset estimate: samples={total_samples or 'unknown'}, batches/epoch~={total_batches}"
    )

    net = ValueNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    start_epoch = 1
    start_step = 0

    print(torch.cuda.is_available(), device)


    if args.resume:
        try:
            loaded_epoch, loaded_step, completed_epoch, _ = load_checkpoint(
                args.resume, net, opt, device
            )
            print(
                f"resumed from {args.resume}: epoch={loaded_epoch} step={loaded_step} completed_epoch={completed_epoch}"
            )
            start_epoch = loaded_epoch + 1
            start_step = loaded_step
            if args.add_epochs and args.add_epochs > 0:
                old_epochs = args.epochs
                args.epochs = (start_epoch - 1) + args.add_epochs
                print(
                    f"adjusting total epochs to {args.epochs} (was {old_epochs}) to run {args.add_epochs} more epochs beyond checkpoint"
                )

            if start_epoch > args.epochs:
                print(
                    f"resume point (epoch {start_epoch}) is beyond requested total epochs ({args.epochs}); nothing to run"
                )
                torch.save({"state_dict": net.state_dict()}, args.out)
                print(f"saved final model {args.out}")
                return

        except FileNotFoundError:
            print(f"resume checkpoint not found: {args.resume}; starting fresh")

    if args.num_workers <= 0:

        if sys.platform == "win32" and torch.cuda.is_available():
            print("Windows with GPU detected; using 0 data loader workers")
            args.num_workers = 0
        else:
            args.num_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"loading preprocessed dataset from {args.preprocessed_dir}")
    ds = PreprocessedDataset(args.preprocessed_dir)

    if args.max_samples and args.max_samples > 0:
        maxs = min(args.max_samples, len(ds))
        print(f"limiting dataset to first {maxs} samples for testing")
        ds = Subset(ds, list(range(maxs)))

    actual_samples = len(ds)
    total_batches = max(1, math.ceil(actual_samples / args.batch))
    print(
        f"using preprocessed dataset: samples={actual_samples}, batches/epoch={total_batches}"
    )


    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    stop_requested = {"flag": False}

    def _signal_handler(signum, frame):
        print(
            f"\nsignal {signum} received, will save checkpoint and exit after finishing current batch"
        )
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    net.train()
    global_step = start_step
    use_amp = args.amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            print(next(net.parameters()).device)
            total = 0.0
            n = 0

            pbar = tqdm(
                dl,
                desc=f"Epoch {epoch}/{args.epochs}",
                unit="batches",
                total=total_batches,
            )

            last_game_id = None
            last_game_sample_idx = None
            for i, batch in enumerate(pbar, start=1):
                if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                    xb, yb, gidb, gidxb = batch
                else:
                    xb, yb = batch
                    gidb = None
                    gidxb = None

                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad()
                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred = net(xb).view(-1, 1)
                        loss = loss_fn(pred, yb)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    pred = net(xb).view(-1, 1)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    opt.step()
                total += float(loss.item())
                n += 1
                global_step += 1

                try:
                    if gidb is not None and gidxb is not None:
                        last_game_id = int(gidb[-1].item())
                        last_game_sample_idx = int(gidxb[-1].item())
                except Exception:
                    last_game_id = None
                    last_game_sample_idx = None

                if i % 10 == 0:
                    running = total / max(n, 1)
                    pbar.set_postfix(loss=f"{running:.6f}",sample_rate=f"{i*args.batch/(time.time()-pbar.start_t):.1f} samples/s")

                if stop_requested["flag"]:
                    pbar.close()
                    print("saving checkpoint before exit...")
                    extra = {}
                    if last_game_id is not None:
                        extra["last_game_id"] = last_game_id
                        extra["last_game_sample_idx"] = last_game_sample_idx or 0
                    try:
                        save_checkpoint(
                            args.checkpoint, net, opt, epoch, global_step, extra=extra
                        )
                        print(f"checkpoint saved to {args.checkpoint}")
                    except Exception:
                        print("failed to save checkpoint")
                    sys.exit(0)

            avg = total / max(n, 1)
            print(f"epoch={epoch} loss={avg:.6f}")

            if args.checkpoint_interval and (epoch % args.checkpoint_interval == 0):
                t0 = time.time()
                save_checkpoint(
                    args.checkpoint,
                    net,
                    opt,
                    epoch,
                    global_step,
                    extra={"completed_epoch": True},
                )
                print(
                    f"saved checkpoint {args.checkpoint} (took {time.time()-t0:.2f}s)"
                )

    except Exception:
        print("unexpected error during training; saving checkpoint before raising")
        try:
            extra = {}
            if "last_game_id" in locals() and last_game_id is not None:
                extra["last_game_id"] = last_game_id
                extra["last_game_sample_idx"] = last_game_sample_idx or 0
            save_checkpoint(args.checkpoint, net, opt, epoch, global_step, extra=extra)
            print(f"checkpoint saved to {args.checkpoint}")
        except Exception:
            print("failed to save checkpoint")
        raise

    torch.save({"state_dict": net.state_dict()}, args.out)
    print(f"saved final model {args.out}")


if __name__ == "__main__":
    main()
