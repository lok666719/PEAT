#!/usr/bin/env python3
"""Evaluate generated adversarial images on a REFER evaluation split."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attacker import evf_sam_attacker


def to_bool_mask(segmentation, original_shape, adversarial_shape):
    segmentation = np.asarray(segmentation).reshape((-1, 2))
    mask = np.zeros(original_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [segmentation.astype(np.int32)], 1)
    mask = cv2.resize(
        mask,
        (adversarial_shape[1], adversarial_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return mask > 0


def iou(mask, target=None):
    mask = np.asarray(mask, dtype=bool)
    if target is None:
        target = np.ones_like(mask, dtype=bool)
        intersection = np.logical_and(~mask, target)
        union = np.logical_or(~mask, target)
    else:
        target = np.asarray(target, dtype=bool)
        intersection = np.logical_and(mask, target)
        union = np.logical_or(mask, target)
    return 100.0 * intersection.sum() / max(union.sum(), 1)


def annotation_at(annotations, annotation_id):
    if isinstance(annotations, dict):
        return annotations.get(annotation_id, annotations.get(str(annotation_id)))
    return annotations[annotation_id]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-pickle", required=True)
    parser.add_argument("--adversarial-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--asr-threshold", type=float, default=0.30)
    return parser


def main():
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the released EVF-SAM implementation.")

    with open(args.test_pickle, "rb") as handle:
        dataset = pickle.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, padding_side="right", use_fast=False
    )
    model = evf_sam_attacker.from_pretrained(
        args.checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model.cuda()
    model.eval()

    adversarial_dir = Path(args.adversarial_dir)
    records = []
    all_foreground_iou = []
    all_background_iou = []
    all_successes = []
    for image in dataset["images"]:
        image_id = image["id"]
        source = Path(image["file_name"])
        adversarial_path = (
            adversarial_dir / f"{image_id}_{source.stem}_adv{source.suffix}"
        )
        adversarial_image = cv2.imread(str(adversarial_path))
        if adversarial_image is None:
            raise FileNotFoundError(adversarial_path)
        adversarial_image = cv2.cvtColor(adversarial_image, cv2.COLOR_BGR2RGB)

        prompts = []
        ground_truths = []
        for ref in dataset["img2refs"].get(image_id, []):
            annotation = annotation_at(dataset["annotations"], ref["ann_id"])
            ground_truth = to_bool_mask(
                annotation["segmentation"][0],
                (image["height"], image["width"]),
                adversarial_image.shape[:2],
            )
            for sentence in ref.get("sentences", []):
                prompts.append(
                    sentence if isinstance(sentence, str) else sentence["sent"]
                )
                ground_truths.append(ground_truth)
        if not prompts:
            continue

        with torch.no_grad():
            masks = model.test(adversarial_image, prompts, tokenizer)[0]
        predictions = masks > 0
        foreground_iou = []
        background_iou = []
        successes = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            prediction = prediction.cpu().numpy().astype(bool)
            ground_truth = np.asarray(ground_truth, dtype=bool)
            foreground_iou.append(iou(prediction, ground_truth))
            background_iou.append(iou(prediction))
            ratio = prediction.sum() / max(ground_truth.sum(), 1)
            successes.append(ratio <= args.asr_threshold)
        all_foreground_iou.extend(foreground_iou)
        all_background_iou.extend(background_iou)
        all_successes.extend(successes)
        records.append(
            {
                "image_id": image_id,
                "num_expressions": len(prompts),
                "ASR": 100.0 * float(np.mean(successes)),
                "mIoU-GT": float(np.mean(foreground_iou)),
                "mIoU-UTM": float(np.mean(background_iou)),
            }
        )

    summary = {
        "num_images": len(records),
        "num_expressions": len(all_successes),
        "ASR": 100.0 * float(np.mean(all_successes)),
        "mIoU-GT": float(np.mean(all_foreground_iou)),
        "mIoU-UTM": float(np.mean(all_background_iou)),
    }
    output = {"summary": summary, "per_image": records}
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
