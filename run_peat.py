#!/usr/bin/env python3
"""Run PEAT or a matched image-only baseline for RES evaluation.

The entry point intentionally keeps the attack implementation in attacker.py
unchanged apart from the configurable proxy-update interval. Tasks may come
from a JSON manifest or from one of the pickle files produced by the dataset
split preparation step.
"""

import argparse
import json
import pickle
import platform
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attacker import evf_sam_attacker


def parse_fraction(value):
    """Accept either a decimal or a fraction such as 16/255."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def prompt_text(sentence):
    if isinstance(sentence, str):
        return sentence
    return sentence["sent"]


def load_tasks(args):
    tasks = []
    if args.task_json:
        with open(args.task_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("tasks", [])
        for item in payload:
            image = item["image"]
            if not Path(image).is_absolute() and args.image_root:
                image = str(Path(args.image_root) / image)
            prompts = [str(prompt) for prompt in item["prompts"]]
            if args.setting == "single-text":
                prompts = prompts[:1]
            tasks.append(
                {
                    "image_id": item.get("image_id", Path(image).stem),
                    "image": image,
                    "prompts": prompts,
                }
            )
    else:
        with open(args.split_pickle, "rb") as handle:
            split = pickle.load(handle)
        image_records = {item["id"]: item for item in split["images"]}
        for image_id, refs in split["img2refs"].items():
            prompts = []
            for ref in refs:
                sentences = ref.get("sentences", [])
                if sentences:
                    prompts.append(prompt_text(sentences[0]))
            if not prompts:
                continue
            if args.setting == "single-text":
                prompts = prompts[:1]
            image = image_records[image_id]["file_name"]
            if not Path(image).is_absolute() and args.image_root:
                image = str(Path(args.image_root) / image)
            tasks.append(
                {
                    "image_id": image_id,
                    "image": image,
                    "prompts": prompts,
                }
            )

    if not tasks:
        raise ValueError("No evaluation tasks were found.")
    return tasks[: args.limit] if args.limit else tasks


def run_method(model, method, task, tokenizer):
    prompts = task["prompts"]
    if method == "peat":
        return model.peat(task["image"], prompts, tokenizer)
    if method == "multi-text":
        return model.multi_prompt_attack(task["image"], prompts, tokenizer)
    if method == "single-text":
        return model.single_prompt_attack(task["image"], prompts[0], tokenizer)
    if method == "segpgd-single":
        return model.segpgd_single_p(task["image"], prompts[0], tokenizer)
    if method == "segpgd-multi":
        return model.seg_pgd_multi_prompt_attack(task["image"], prompts, tokenizer)
    raise ValueError(f"Unsupported method: {method}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate adversarial RES images with PEAT or a baseline."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task-json", help="JSON list of image and prompt tasks.")
    source.add_argument("--split-pickle", help="Optimization split pickle file.")
    parser.add_argument(
        "--image-root",
        default="",
        help="Prefix for relative image paths in the task source.",
    )
    parser.add_argument(
        "--setting",
        choices=("single-text", "multi-text"),
        default="multi-text",
        help="Use one prompt or one prompt per annotated object.",
    )
    parser.add_argument("--checkpoint", required=True, help="EVF-SAM checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for adversarial images.")
    parser.add_argument(
        "--method",
        choices=(
            "peat",
            "single-text",
            "multi-text",
            "segpgd-single",
            "segpgd-multi",
        ),
        default="peat",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--proxy-update-interval", type=int, default=10)
    parser.add_argument("--epsilon", default="16/255")
    parser.add_argument("--step-size", default="2/255")
    parser.add_argument("--embedding-budget", type=float, default=0.2)
    parser.add_argument("--embedding-step-size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--efficiency-report",
        default="efficiency.json",
        help="Path for runtime and GPU-memory measurements.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the released EVF-SAM implementation.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        padding_side="right",
        use_fast=False,
    )
    model = evf_sam_attacker.from_pretrained(
        args.checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model.cuda()
    model.eval()

    model.adv_iters = args.iterations
    model.proxy_update_interval = args.proxy_update_interval
    model.adv_epsilon = parse_fraction(args.epsilon) * 255.0
    model.adv_alpha = parse_fraction(args.step_size) * 255.0
    model.adv_epsilon_text = args.embedding_budget
    model.adv_alpha_text = args.embedding_step_size

    tasks = load_tasks(args)
    measurements = []
    for index, task in enumerate(tasks, start=1):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        adversarial_image, losses = run_method(model, args.method, task, tokenizer)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        source = Path(task["image"])
        output_path = output_dir / f"{task['image_id']}_{source.stem}_adv{source.suffix}"
        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(adversarial_image, cv2.COLOR_RGB2BGR),
        )
        measurements.append(
            {
                "image_id": task["image_id"],
                "image": task["image"],
                "prompt_count": len(task["prompts"]),
                "output": str(output_path),
                "seconds": elapsed,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
                "final_loss": losses[-1] if losses else None,
            }
        )
        print(f"[{index}/{len(tasks)}] {task['image_id']}: {elapsed:.3f}s")
        torch.cuda.empty_cache()

    seconds = [item["seconds"] for item in measurements]
    report = {
        "method": args.method,
        "setting": args.setting,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "configuration": {
            "iterations": args.iterations,
            "proxy_update_interval": args.proxy_update_interval,
            "epsilon": args.epsilon,
            "step_size": args.step_size,
            "embedding_budget": args.embedding_budget,
            "embedding_step_size": args.embedding_step_size,
            "seed": args.seed,
        },
        "summary": {
            "num_images": len(measurements),
            "mean_seconds_per_image": sum(seconds) / len(seconds),
            "total_seconds": sum(seconds),
            "total_hours": sum(seconds) / 3600.0,
            "peak_allocated_gb": max(
                item["peak_allocated_gb"] for item in measurements
            ),
            "peak_reserved_gb": max(
                item["peak_reserved_gb"] for item in measurements
            ),
        },
        "measurements": measurements,
    }
    report_path = Path(args.efficiency_report)
    if not report_path.is_absolute():
        report_path = output_dir / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in report.items() if key != "measurements"}, handle, indent=2)
    print(f"Efficiency report: {report_path}")


if __name__ == "__main__":
    main()
