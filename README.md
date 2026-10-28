# Cross-Text Transferable Adversarial Attacks on Referring Expression Segmentation via Proxy Embedding Guidance

## Version

This source tree is the `v1.1.1` reproducibility revision prepared for the
third-round manuscript. It builds on the publicly archived `v1.1.0`
release (commit `c4efc6c`).

The `v1.1.0` version-specific archive is
https://doi.org/10.5281/zenodo.21930248. The concept DOI
https://doi.org/10.5281/zenodo.17445787 identifies all archived versions.

## Description
We propose **Proxy-Embedding as an Adversarial Teacher (PEAT)** — an embedding-guided bidirectional adversarial attack framework for Referring Expression Segmentation (RES) models. The method crafts cross-text transferable adversarial examples by leveraging proxy embeddings as an adversarial teacher. 

## Datasets and Data Splits

We follow the dataset preparation conventions of EVF-SAM for
**RefCOCO**, **RefCOCO+**, and **RefCOCOg**. Please obtain the datasets
from the original sources and follow their corresponding licenses and
usage requirements.

Dataset preparation instructions can be found in the official EVF-SAM
repository:

- **RefCOCO / RefCOCO+ / RefCOCOg**  
  https://github.com/hustvl/evf-sam

The reproducible dataset splitting utility is
[`prepare_splits.py`](prepare_splits.py).

`split_manifest.json` records the sampling rules, random seeds, image
budget, and optimization/evaluation-expression definitions. The original
datasets and pretrained weights are not redistributed. Dataset-specific
image IDs are therefore reconstructed from the official annotations with
the documented seeds and split settings rather than distributed as
third-party data.

For example, generate the RefCOCO split for seed 42:

```bash
python prepare_splits.py \
  --data-root path/to/refer_seg \
  --dataset refcoco \
  --split-by unc \
  --split testA \
  --seed 42 \
  --num-images 750 \
  --output-dir splits/refcoco/seed_42
```

The command writes `train.p`, `test.p`, and a seed-specific
`split-manifest.json` containing the sampled image IDs, annotation IDs,
optimization expressions, and evaluation expressions.
If the official `refer.py` module is not already on `PYTHONPATH`, provide
its directory with `--refer-path`.

> **Note:** Some datasets may require permission, registration, or
> manual download from their original providers. We do not redistribute
> the original datasets in this repository.

- **RefCOCO / RefCOCO+ / RefCOCOg**  
  Download / prepare following EVF-SAM instructions: https://github.com/hustvl/evf-sam?tab=readme-ov-file#-early-vision-language-fusion-for-text-prompted-segment-anything-model-
  
## RES Models and Pretrained Checkpoints

PEAT is evaluated on five RES model configurations: **EVF-SAM**, 
**EVF-SAM2**, **DMMI with a ResNet backbone**, **DMMI with a Swin 
Transformer backbone**, and **LAVT**.

The command-line runner `run_peat.py` in this repository is provided for
the EVF-SAM implementation. The other model configurations require the
corresponding official model implementations and checkpoints listed
below. Model weights are not redistributed.

We do not redistribute pretrained model weights in this repository. 
Please obtain the official model implementations and pretrained 
checkpoints from the corresponding original repositories:

- **EVF-SAM / EVF-SAM2**  
  Official repository:  
  https://github.com/hustvl/evf-sam

- **DMMI (ResNet / Swin Transformer)**  
  Official repository:  
  https://github.com/toggle1995/RIS-DMMI

- **LAVT**  
  Official repository:  
  https://github.com/yz93/LAVT-RIS

Please follow the instructions provided in the corresponding repositories
to download the pretrained checkpoints and prepare the model
configurations.

A recommended local directory structure is:

```text
checkpoints/
├── evf_sam/
├── evf_sam2/
├── dmmi_resnet/
├── dmmi_swin/
└── lavt/
```

> Note: Some datasets or model weights may require permission or registration from their original authors — follow the instructions in those repositories.

## Cross-Text RES Robustness Evaluation Protocol

In addition to the PEAT attack method, this repository provides a
reproducible evaluation protocol for studying cross-text adversarial
robustness in Referring Expression Segmentation (RES).

The released protocol includes:

- deterministic image-sampling and expression-selection rules;
- strictly disjoint optimization and evaluation expressions;
- Single-Text and Multi-Text evaluation settings;
- baseline attack implementations;
- model and attack configurations;
- scripts for adversarial-example generation and evaluation; and
- reproducibility information for the experiments reported in the paper.

The sampling rules used in the paper are provided in
`split_manifest.json`, and the splitting procedure is implemented in
[`prepare_splits.py`](prepare_splits.py).
Exact image and expression identifiers are not bundled because the
original dataset images and annotations are not redistributed. They can
be reconstructed from the official annotations using the released seed,
split, and sampling parameters.

The protocol is intended to provide a common and reusable basis for
evaluating whether adversarial images generated using a limited set of
referring expressions remain effective under strictly unseen textual
inputs.

## Requirement

Recommended environment: Linux, NVIDIA A100 GPU, CUDA 11.7, and Python
3.9. Install the tested dependencies with:

```bash
pip install --index-url https://download.pytorch.org/whl/cu117 \
  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```

## Troubleshooting

- **CUDA or PyTorch version mismatch:** Please ensure that PyTorch is
  installed with CUDA 11.7 as specified in the environment file.

- **Checkpoint not found:** Verify that the directory supplied to
  `--checkpoint` contains the EVF-SAM configuration, tokenizer, and model
  weights.

- **Dataset path error:** Check that the RefCOCO/RefCOCO+/RefCOCOg
  annotations and images follow the directory structure described in
  the dataset preparation section.

- **Out-of-memory error:** Reduce the batch size where applicable and
  ensure sufficient GPU memory is available.

- **Different evaluation results:** Verify that the provided fixed split
  parameters, official dataset split, and random seed are being used.

## Attack and Evaluate

The task source can be a JSON manifest or an optimization split pickle
following the REFER structure. A JSON manifest has the form:

```json
[
  {
    "image_id": 123,
    "image": "images/COCO_train2014_000000000123.jpg",
    "prompts": ["the person on the left", "the child in a red coat"]
  }
]
```

For the generated optimization split, replace `--task-json tasks.json`
with `--split-pickle splits/refcoco/seed_42/train.p`.

Generate adversarial images with PEAT:

```bash
python run_peat.py \
  --checkpoint checkpoints/evf_sam \
  --task-json tasks.json \
  --image-root path/to/refcoco \
  --setting multi-text \
  --output-dir results/peat
```

Run a protocol-matched image-only baseline with the same task manifest:

```bash
python run_peat.py \
  --checkpoint checkpoints/evf_sam \
  --task-json tasks.json \
  --setting multi-text \
  --method multi-text \
  --output-dir results/multi_text
```

Evaluate the generated adversarial images with the held-out expressions:

```bash
python evaluate_peat.py \
  --checkpoint checkpoints/evf_sam \
  --test-pickle splits/refcoco/seed_42/test.p \
  --adversarial-dir results/peat \
  --output-json results/peat/metrics.json
```

The default attack configuration is 100 iterations, epsilon
`16/255`, step size `2/255`, proxy-embedding budget `0.2`, proxy step
size `0.01`, and proxy-update interval `10`, matching the manuscript.

## Runtime and GPU Memory

`run_peat.py` records per-image wall-clock generation time and PyTorch
peak allocated/reserved GPU memory. The aggregate report is written to
`efficiency.json` in the output directory:

```bash
python run_peat.py \
  --checkpoint checkpoints/evf_sam \
  --task-json tasks.json \
  --image-root path/to/refcoco \
  --method peat \
  --output-dir results/efficiency \
  --efficiency-report efficiency.json
```

Timing excludes dataset loading and model initialization and includes
one complete adversarial-image generation call. Memory is reported after
model loading as the maximum CUDA allocation observed during the call.

The original `adv_eval.py`, `adv_eval_test.py`, and
`utils/splite_dataset_new_random.py` are retained as legacy experiment
scripts. The commands documented above use the portable entry points
`prepare_splits.py`, `run_peat.py`, and `evaluate_peat.py`.
