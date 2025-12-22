# src/infer.py
import os
import glob
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import pandas as pd
from tqdm import tqdm


class TestDataset(Dataset):
    def __init__(self, folder: str, transform=None):
        self.paths = sorted(glob.glob(os.path.join(folder, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(p)


def load_classes(classes_txt_path: str):
    with open(classes_txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_model(num_classes: int):
    # In inference we load weights from checkpoint; no need to download ImageNet weights.
    model = efficientnet_b0(weights=None)
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_feats, num_classes)
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset root containing test/ folder.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pth")
    parser.add_argument("--classes", type=str, default="outputs/classes.txt",
                        help="Path to classes.txt saved during training")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_csv", type=str, default="outputs/submission.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = load_classes(args.classes)
    num_classes = len(classes)

    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_folder = os.path.join(args.data_dir, "test")
    test_ds = TestDataset(folder=test_folder, transform=val_test_transform)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model(num_classes).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    predictions = []
    with torch.no_grad():
        for imgs, filenames in tqdm(test_loader, desc="Infer"):
            imgs = imgs.to(device, non_blocking=True)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            preds = preds.cpu().numpy()
            for f, p in zip(filenames, preds):
                predictions.append((f, classes[int(p)]))

    df = pd.DataFrame(predictions, columns=["filename", "label"])
    df = df.sort_values("filename").reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved predictions to: {args.out_csv}")


if __name__ == "__main__":
    main()
