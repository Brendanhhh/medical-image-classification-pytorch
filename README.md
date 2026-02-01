Medical Image Classification (PyTorch)

Multi-class medical image classification using EfficientNet-B0 transfer learning in PyTorch.
Developed for the Kaggle competition msu-cse-404-fs-25-project.

Highlights

Model: EfficientNet-B0 (ImageNet pretrained)

Training: AMP mixed precision, AdamW optimizer, cosine annealing warm restarts

Augmentation: Random resized crop, horizontal flip, rotation, color jitter

Validation Accuracy: ~0.83 on held-out validation set (varies by run)

Package Management: uv

Task Definition

This project performs single-label, multi-class image classification on a medical imaging dataset with anonymized class labels.

Each image is assigned to one of ~21 dataset-defined categories

Class labels are represented as opaque alphanumeric identifiers

No mapping from label → diagnosis, condition, or modality is provided

As a result, the model learns to distinguish visual patterns associated with dataset-specific categories, rather than predicting human-interpretable medical conditions. The project emphasizes model design, training stability, and generalization, not clinical inference.

## Project Structure

.
├── src/
│   ├── train.py        # training script
│   └── infer.py        # inference / submission generation
├── outputs/            # model checkpoints and predictions (not tracked)
├── main.py             # optional orchestration entrypoint
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md

## Dataset (not included)

This repository does NOT include the dataset.

The data comes from the Kaggle competition:

Competition: msu-cse-404-fs-25-project

### Download the dataset

1. Install and configure the Kaggle CLI:

pip install kaggle

Place your kaggle.json API token in:
- Windows: %USERPROFILE%\.kaggle\kaggle.json
- macOS / Linux: ~/.kaggle/kaggle.json

2. Download and unzip the competition data:

kaggle competitions download -c msu-cse-404-fs-25-project
unzip msu-cse-404-fs-25-project.zip

This will create the dataset directory:

medical_image_dataset/
├── train/
│   ├── <class_name>/*.png
├── validation/
│   ├── <class_name>/*.png
└── test/
    └── *.png

## Training

Train the model and save the best checkpoint to the outputs/ directory:

uv run src/train.py --data_dir medical_image_dataset --out_dir outputs --epochs 10 --batch_size 64

The best model checkpoint is saved to:

outputs/best_model.pth

A class label mapping is saved to:

outputs/classes.txt

## Inference / Submission Generation

Generate predictions for the test set:

uv run src/infer.py --data_dir medical_image_dataset --checkpoint outputs/best_model.pth --out_csv outputs/submission.csv

The output CSV contains:

filename,label

## Model Details

- Backbone: EfficientNet-B0
- Pretrained weights: ImageNet (training only)
- Input size: 224 x 224
- Loss: Cross-Entropy
- Optimizer: AdamW
- Scheduler: CosineAnnealingWarmRestarts
- Mixed precision training enabled when using CUDA

## Requirements
- Python >= 3.10
- uv (https://docs.astral.sh/uv/)

## Installation

This project uses uv for fast and reproducible dependency management.

1. Install uv (if not already installed):

pip install uv

2. Sync dependencies (creates .venv and installs all locked packages):

uv sync

> All commands above use `uv run`.  
> Plain `python` can also be used if the virtual environment is activated manually.

## Notes

- The dataset is excluded due to Kaggle competition rules.
- outputs/ is intentionally not tracked in version control.
- Test images are assumed to be PNG files.
