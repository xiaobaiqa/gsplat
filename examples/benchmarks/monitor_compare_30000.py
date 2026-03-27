import argparse
import json
import os
from typing import Dict, List


SCENES = ["bicycle", "stump", "bonsai", "counter", "kitchen", "room"]


def _try_load(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _scene_metrics(root: str, scene: str, final_step: int):
    stats_dir = os.path.join(root, scene, "stats")
    train = _try_load(os.path.join(stats_dir, f"train_step{final_step}_rank0.json"))
    val = _try_load(os.path.join(stats_dir, f"val_step{final_step}.json"))
    if train is None or val is None:
        return None
    return {
        "psnr": float(val["psnr"]),
        "ssim": float(val["ssim"]),
        "lpips": float(val["lpips"]),
        "num_GS": int(val.get("num_GS", train.get("num_GS", -1))),
        "train_mem_gb": float(train["mem"]),
        "train_time_s": float(train["ellipse_time"]),
    }


def _mean(rows: List[Dict], key: str) -> float:
    return sum(r[key] for r in rows) / max(1, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", default="results/benchmark_15000_all")
    parser.add_argument("--rc-root", default="results/benchmark_rc_15000_all")
    parser.add_argument("--log", default="results/compare_30000_all_scenes.log")
    parser.add_argument("--final-step", type=int, default=14999)
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    args = parser.parse_args()

    started = []
    if os.path.isfile(args.log):
        with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("[RUN] "):
                    started.append(line.strip())

    baseline_rows: List[Dict] = []
    rc_rows: List[Dict] = []
    delta_rows: List[Dict] = []

    print("=== Per-scene status ===")
    for scene in args.scenes:
        b = _scene_metrics(args.baseline_root, scene, args.final_step)
        r = _scene_metrics(args.rc_root, scene, args.final_step)
        b_ok = b is not None
        r_ok = r is not None
        print(f"{scene:8s} baseline={'OK' if b_ok else '...'} rc={'OK' if r_ok else '...'}")

        if b_ok:
            baseline_rows.append(b)
        if r_ok:
            rc_rows.append(r)
        if b_ok and r_ok:
            delta_rows.append(
                {
                    "psnr": r["psnr"] - b["psnr"],
                    "ssim": r["ssim"] - b["ssim"],
                    "lpips": r["lpips"] - b["lpips"],
                    "num_GS": r["num_GS"] - b["num_GS"],
                    "train_mem_gb": r["train_mem_gb"] - b["train_mem_gb"],
                    "train_time_s": r["train_time_s"] - b["train_time_s"],
                }
            )

    print("\n=== Progress ===")
    print(f"started jobs markers: {len(started)}")
    if started:
        print(f"latest marker: {started[-1]}")
    print(f"baseline done scenes: {len(baseline_rows)}/{len(args.scenes)}")
    print(f"rc done scenes: {len(rc_rows)}/{len(args.scenes)}")
    print(f"paired done scenes: {len(delta_rows)}/{len(args.scenes)}")

    if delta_rows:
        print("\n=== Paired mean delta (rc - baseline) ===")
        print(f"PSNR      {_mean(delta_rows, 'psnr'):.4f}")
        print(f"SSIM      {_mean(delta_rows, 'ssim'):.4f}")
        print(f"LPIPS     {_mean(delta_rows, 'lpips'):.4f}")
        print(f"num_GS    {_mean(delta_rows, 'num_GS'):.1f}")
        print(f"mem(GB)   {_mean(delta_rows, 'train_mem_gb'):.4f}")
        print(f"time(s)   {_mean(delta_rows, 'train_time_s'):.2f}")


if __name__ == "__main__":
    main()