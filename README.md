# Medical Image Classification (PyTorch)

Multi-class medical image classification using EfficientNet-B0 transfer learning in PyTorch.

## Highlights
- Model: EfficientNet-B0 (ImageNet pretrained)
- Training: AMP mixed precision, AdamW optimizer, cosine annealing warm restarts
- Augmentation: random resized crop, flip, rotation, color jitter
- Best validation accuracy: ~0.832 (dataset-specific)

## Repo contents
- `src/train.py`: train and save best checkpoint
- `src/infer.py`: run inference on a folder of test images and write `submission.csv`

## Run (example)
```bash
python src/train.py --data_dir /path/to/medical_image_dataset --epochs 10 --batch_size 64
python src/infer.py --data_dir /path/to/medical_image_dataset --checkpoint outputs/best_model.pth
