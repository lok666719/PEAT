#!/usr/bin/env python3
"""Create deterministic RES optimization/evaluation splits from REFER data."""

import argparse
import copy
import json
import os
import pickle
import random
import sys
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="REFER dataset root.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("refcoco", "refcoco+", "refcocog"),
    )
    parser.add_argument("--split-by", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-images", type=int, default=750)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--refer-path",
        default="",
        help="Optional directory containing the official refer.py module.",
    )
    return parser


def sentence_text(sentence):
    return sentence if isinstance(sentence, str) else sentence["sent"]


def main():
    args = build_parser().parse_args()
    if args.refer_path:
        sys.path.insert(0, args.refer_path)
    from refer import REFER

    random.seed(args.seed)
    refer_api = REFER(args.data_root, args.dataset, args.split_by)
    all_image_ids = sorted(refer_api.getImgIds(ref_ids=refer_api.getRefIds(split=args.split)))
    if len(all_image_ids) < args.num_images:
        raise ValueError(
            f"Requested {args.num_images} images, but only {len(all_image_ids)} are available."
        )
    image_ids = random.sample(all_image_ids, args.num_images)
    selected = set(image_ids)
    refs = [
        ref
        for ref in refer_api.loadRefs(ref_ids=refer_api.getRefIds(split=args.split))
        if ref["image_id"] in selected
    ]

    images = []
    for item in refer_api.loadImgs(image_ids=image_ids):
        item = item.copy()
        item["file_name"] = os.path.join(
            args.data_root,
            "images",
            "mscoco",
            "images",
            "train2014",
            item["file_name"],
        )
        images.append(item)

    img2refs = {}
    for ref in refs:
        img2refs.setdefault(ref["image_id"], []).append(ref)
    source = {"images": images, "annotations": refer_api.Anns, "img2refs": img2refs}
    optimization = copy.deepcopy(source)
    evaluation = copy.deepcopy(source)

    records = []
    for image_id in image_ids:
        image_records = []
        for object_index, ref in enumerate(img2refs.get(image_id, [])):
            sentences = list(ref.get("sentences", []))
            if not sentences:
                continue
            optimization_index = random.randrange(len(sentences))
            optimization_sentence = sentences[optimization_index]
            evaluation_sentences = (
                sentences[:optimization_index] + sentences[optimization_index + 1 :]
            )
            optimization["img2refs"][image_id][object_index]["sentences"] = [
                optimization_sentence
            ]
            evaluation["img2refs"][image_id][object_index]["sentences"] = (
                evaluation_sentences
            )
            image_records.append(
                {
                    "annotation_id": ref.get("ann_id"),
                    "optimization_expression": sentence_text(optimization_sentence),
                    "evaluation_expressions": [
                        sentence_text(sentence) for sentence in evaluation_sentences
                    ],
                }
            )
        records.append({"image_id": image_id, "annotations": image_records})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("train.p", optimization), ("test.p", evaluation)):
        with open(output_dir / name, "wb") as handle:
            pickle.dump(payload, handle)

    manifest = {
        "dataset": args.dataset,
        "split_by": args.split_by,
        "split": args.split,
        "seed": args.seed,
        "num_images": len(records),
        "settings": ["single_text", "multi_text"],
        "single_text_rule": "first optimization expression",
        "multi_text_rule": "one optimization expression per annotated object",
        "evaluation_rule": "all remaining expressions",
        "images": records,
    }
    with open(output_dir / "split-manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote {len(records)} image splits to {output_dir}")


if __name__ == "__main__":
    main()
