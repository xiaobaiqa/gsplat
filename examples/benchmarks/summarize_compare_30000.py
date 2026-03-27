import argparse
import json
import os
from typing import Dict, List


SCENES = ["bicycle", "stump", "bonsai", "counter", "kitchen", "room"]


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_scene_stats(root: str, scene: str, final_step: int) -> Dict:
    stats_dir = os.path.join(root, scene, "stats")
    train_path = os.path.join(stats_dir, f"train_step{final_step}_rank0.json")
    val_path = os.path.join(stats_dir, f"val_step{final_step}.json")
    if not (os.path.isfile(train_path) and os.path.isfile(val_path)):
        return {"ok": False, "missing": [train_path, val_path]}

    train = _read_json(train_path)
    val = _read_json(val_path)
    return {
        "ok": True,
        "train": train,
        "val": val,
        "metrics": {
            "psnr": float(val["psnr"]),
            "ssim": float(val["ssim"]),
            "lpips": float(val["lpips"]),
            "num_GS": int(val.get("num_GS", train.get("num_GS", -1))),
            "train_mem_gb": float(train["mem"]),
            "train_time_s": float(train["ellipse_time"]),
        },
        "paths": {
            "train": train_path,
            "val": val_path,
        },
    }


def _avg(rows: List[Dict], key: str) -> float:
    return sum(r[key] for r in rows) / max(len(rows), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--rc-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--final-step", type=int, default=14999)
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    args = parser.parse_args()

    results = {
        "baseline_root": args.baseline_root,
        "rc_root": args.rc_root,
        "final_step": args.final_step,
        "scenes": {},
        "aggregate": {},
    }

    valid_baseline = []
    valid_rc = []
    valid_delta = []

    for scene in args.scenes:
        b = _load_scene_stats(args.baseline_root, scene, args.final_step)
        r = _load_scene_stats(args.rc_root, scene, args.final_step)
        row = {"baseline": b, "residual_coverage": r}

        if b["ok"] and r["ok"]:
            delta = {
                "psnr": r["metrics"]["psnr"] - b["metrics"]["psnr"],
                "ssim": r["metrics"]["ssim"] - b["metrics"]["ssim"],
                "lpips": r["metrics"]["lpips"] - b["metrics"]["lpips"],
                "num_GS": r["metrics"]["num_GS"] - b["metrics"]["num_GS"],
                "train_mem_gb": r["metrics"]["train_mem_gb"] - b["metrics"]["train_mem_gb"],
                "train_time_s": r["metrics"]["train_time_s"] - b["metrics"]["train_time_s"],
            }
            row["delta_rc_minus_baseline"] = delta
            valid_baseline.append(b["metrics"])
            valid_rc.append(r["metrics"])
            valid_delta.append(delta)

        results["scenes"][scene] = row

    if valid_baseline and valid_rc:
        results["aggregate"] = {
            "n_valid_scenes": len(valid_baseline),
            "baseline_mean": {
                "psnr": _avg(valid_baseline, "psnr"),
                "ssim": _avg(valid_baseline, "ssim"),
                "lpips": _avg(valid_baseline, "lpips"),
                "num_GS": _avg(valid_baseline, "num_GS"),
                "train_mem_gb": _avg(valid_baseline, "train_mem_gb"),
                "train_time_s": _avg(valid_baseline, "train_time_s"),
            },
            "residual_coverage_mean": {
                "psnr": _avg(valid_rc, "psnr"),
                "ssim": _avg(valid_rc, "ssim"),
                "lpips": _avg(valid_rc, "lpips"),
                "num_GS": _avg(valid_rc, "num_GS"),
                "train_mem_gb": _avg(valid_rc, "train_mem_gb"),
                "train_time_s": _avg(valid_rc, "train_time_s"),
            },
            "delta_mean_rc_minus_baseline": {
                "psnr": _avg(valid_delta, "psnr"),
                "ssim": _avg(valid_delta, "ssim"),
                "lpips": _avg(valid_delta, "lpips"),
                "num_GS": _avg(valid_delta, "num_GS"),
                "train_mem_gb": _avg(valid_delta, "train_mem_gb"),
                "train_time_s": _avg(valid_delta, "train_time_s"),
            },
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()