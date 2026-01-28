# Medical Image Classification (PyTorch)

Multi-class medical image classification using EfficientNet-B0 transfer learning in PyTorch.

## Highlights
- **Model:** EfficientNet-B0 (ImageNet pretrained)
- **Training:** AMP mixed precision, AdamW optimizer, cosine annealing warm restarts
- **Augmentation:** Random resized crop, flip, rotation, color jitter
- **Best Validation Accuracy:** ~0.832 (dataset-specific)
- **Package Management:** [uv](https://github.com/astral-sh/uv)

## Project Structure
The repository is organized as follows:

- `src/`: Source code for training and inference modules.
- `main.py`: Primary entry point script.
- `outputs/`: Directory for saved model checkpoints and logs.
- `pyproject.toml` & `uv.lock`: Project configuration and dependency lockfiles.
- `requirements.txt`: Legacy dependency list.

## Requirements
- Python >= 3.14
- [uv](https://docs.astral.sh/uv/)

## Installation

This project uses `uv` for extremely fast dependency management.

1. **Install uv** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Sync dependencies**:
   This will create the virtual environment (`.venv`) and install all lock-file dependencies (Torch, Pandas, Pillow, etc.).
   ```bash
   uv sync
   ```

## Usage

You can run scripts using `uv run` to automatically utilize the project's virtual environment.

### Training
Train the model and save the best checkpoint to the `outputs/` directory.

```bash
uv run src/train.py --data_dir /path/to/medical_image_dataset --epochs 10 --batch_size 64
```

### Inference
Run inference on a folder of test images to generate a `submission.csv`.

```bash
uv run src/infer.py --data_dir /path/to/medical_image_dataset --checkpoint outputs/best_model.pth
```

### Main Script
If `main.py` acts as your central orchestrator, run it directly:

```bash
uv run main.py
```