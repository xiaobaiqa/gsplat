import argparse
import json
import os
from typing import Dict, List


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_stats(root: str, final_step: int) -> Dict:
    stats_dir = os.path.join(root, "garden", "stats")
    train_path = os.path.join(stats_dir, f"train_step{final_step}_rank0.json")
    val_path = os.path.join(stats_dir, f"val_step{final_step}.json")
    if not (os.path.isfile(train_path) and os.path.isfile(val_path)):
        return {
            "ok": False,
            "root": root,
            "missing": [train_path, val_path],
        }

    train = _read_json(train_path)
    val = _read_json(val_path)
    return {
        "ok": True,
        "root": root,
        "metrics": {
            "psnr": float(val["psnr"]),
            "ssim": float(val["ssim"]),
            "lpips": float(val["lpips"]),
            "num_GS": int(val.get("num_GS", train.get("num_GS", -1))),
            "train_mem_gb": float(train["mem"]),
            "train_time_s": float(train["ellipse_time"]),
        },
    }


def _delta(cur: Dict, baseline: Dict) -> Dict:
    return {
        "psnr": cur["psnr"] - baseline["psnr"],
        "ssim": cur["ssim"] - baseline["ssim"],
        "lpips": cur["lpips"] - baseline["lpips"],
        "num_GS": cur["num_GS"] - baseline["num_GS"],
        "num_GS_ratio": cur["num_GS"] / baseline["num_GS"],
        "train_mem_gb": cur["train_mem_gb"] - baseline["train_mem_gb"],
        "train_time_s": cur["train_time_s"] - baseline["train_time_s"],
        "train_time_ratio": cur["train_time_s"] / baseline["train_time_s"],
    }


def _print_table(rows: List[Dict]) -> None:
    header = (
        f"{'name':<28} {'psnr':>8} {'ssim':>8} {'lpips':>8} "
        f"{'num_GS':>10} {'mem':>8} {'time':>8} {'gs_ratio':>8} {'time_ratio':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        metrics = row["metrics"]
        delta = row.get("delta")
        gs_ratio = "-" if delta is None else f"{delta['num_GS_ratio']:.3f}"
        time_ratio = "-" if delta is None else f"{delta['train_time_ratio']:.3f}"
        print(
            f"{row['name']:<28} "
            f"{metrics['psnr']:>8.4f} "
            f"{metrics['ssim']:>8.4f} "
            f"{metrics['lpips']:>8.4f} "
            f"{metrics['num_GS']:>10d} "
            f"{metrics['train_mem_gb']:>8.2f} "
            f"{metrics['train_time_s']:>8.2f} "
            f"{gs_ratio:>8} "
            f"{time_ratio:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--candidate-root", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--final-step", type=int, default=14999)
    args = parser.parse_args()

    baseline = _load_stats(args.baseline_root, args.final_step)
    if not baseline["ok"]:
        raise FileNotFoundError(f"Missing baseline stats: {baseline['missing']}")

    rows = [
        {
            "name": os.path.basename(args.baseline_root.rstrip("/")),
            "root": args.baseline_root,
            "metrics": baseline["metrics"],
            "delta": None,
        }
    ]
    results = {"baseline": baseline, "candidates": []}

    for root in args.candidate_root:
        item = _load_stats(root, args.final_step)
        item["name"] = os.path.basename(root.rstrip("/"))
        if item["ok"]:
            item["delta"] = _delta(item["metrics"], baseline["metrics"])
            rows.append(
                {
                    "name": item["name"],
                    "root": root,
                    "metrics": item["metrics"],
                    "delta": item["delta"],
                }
            )
        results["candidates"].append(item)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    _print_table(rows)


if __name__ == "__main__":
    main()
