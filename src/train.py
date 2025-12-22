# src/train.py
import os
import copy
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm


def build_transforms(img_size: int):
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def build_loaders(data_dir: str, img_size: int, batch_size: int, num_workers: int, pin_memory: bool):
    train_tf, val_tf = build_transforms(img_size)

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "validation")

    train_ds = datasets.ImageFolder(root=train_path, transform=train_tf)
    val_ds = datasets.ImageFolder(root=val_path, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_ds, val_ds, train_loader, val_loader


def build_model(num_classes: int):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_feats, num_classes)
    )
    return model


def train_one_epoch(model, loader, optimizer, criterion, device, use_amp: bool, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{running_loss / max(total, 1):.4f}",
            "acc": f"{correct / max(total, 1):.4f}"
        })

    return running_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "val_loss": f"{running_loss / max(total, 1):.4f}",
            "val_acc": f"{correct / max(total, 1):.4f}"
        })

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset root containing train/ validation/ test/ folders.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--t0", type=int, default=3, help="CosineAnnealingWarmRestarts T_0")
    parser.add_argument("--t_mult", type=int, default=1, help="CosineAnnealingWarmRestarts T_mult")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="CosineAnnealingWarmRestarts eta_min")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    train_ds, val_ds, train_loader, val_loader = build_loaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    num_classes = len(train_ds.classes)
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=args.eta_min
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, use_amp, scaler)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)
            print(f"New best model saved: {ckpt_path}")

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    # (Optional) ensure final model in memory is best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save class mapping for inference convenience
    classes_path = os.path.join(args.out_dir, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        for c in train_ds.classes:
            f.write(c + "\n")
    print(f"Saved class list: {classes_path}")


if __name__ == "__main__":
    main()
