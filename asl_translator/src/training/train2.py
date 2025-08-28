import os
import sys
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_lstm import ASLTranslator, ASLDataLoader
from models.resnet50_bilstm import ResNet50BiLSTM


# -----------------------------
# Dataset
# -----------------------------
class ASLDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        self.targets = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for video_file in os.listdir(class_dir):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    self.samples.append((os.path.join(class_dir, video_file), class_name))
                    self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        video_path, class_name = self.samples[idx]
        loader = ASLDataLoader(video_path, self.transform)
        frames = loader.load_video()  # expected shape: [T, C, H, W]
        label = self.class_to_idx[class_name]
        return frames, label


# -----------------------------
# Collate (pad 1st and last dims)
# -----------------------------
def collate_fn_padd(batch):
    # Max sequence length (dim 0) and width (last dim) across batch
    max_len = max([item[0].size(0) for item in batch])
    max_last_dim = max([item[0].size(-1) for item in batch])

    padded_inputs, labels = [], []
    for inputs, label in batch:
        padding_size_seq = max_len - inputs.size(0)
        padding_size_last_dim = max_last_dim - inputs.size(-1)
        padded_input = torch.nn.functional.pad(
            inputs, (0, padding_size_last_dim, 0, 0, 0, 0, 0, padding_size_seq)
        )
        padded_inputs.append(padded_input)
        labels.append(label)
    return torch.stack(padded_inputs), torch.tensor(labels)


# -----------------------------
# Utils for overfit harness
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_balanced_subset(dataset, n: int, seed: int = 1337):
    """
    Choose up to n indices, round-robin over classes.
    Works with:
      - custom dataset exposing .targets
      - any dataset where dataset[i] -> (x, y) with int y
    """
    rng = random.Random(seed)

    labels = None
    # Fast path: direct targets
    if hasattr(dataset, "targets") and dataset.targets is not None and len(dataset.targets) == len(dataset):
        labels = list(dataset.targets)
    else:
        # Fall back to probing (handles torch.utils.data.Subset, etc.)
        try:
            labels = [int(dataset[i][1]) for i in range(len(dataset))]
        except Exception:
            # Fallback: no labels accessible -> first n
            return list(range(min(n, len(dataset))))

    buckets = defaultdict(list)
    for idx, y in enumerate(labels):
        buckets[int(y)].append(idx)

    order = sorted(buckets.keys())
    chosen = []
    while len(chosen) < min(n, len(dataset)):
        progressed = False
        for c in order:
            if buckets[c]:
                chosen.append(buckets[c].pop(rng.randrange(len(buckets[c]))))
                progressed = True
                if len(chosen) == n:
                    break
        if not progressed:
            break
    return chosen


def apply_dropout_override(model: nn.Module, p: float):
    import torch.nn as nn
    def set_drop(m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.p = p
    model.apply(set_drop)


def build_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "resnet50_bilstm":
        return ResNet50BiLSTM(num_classes=num_classes)
    elif name == "cnn_lstm":
        # ASLTranslator signature may differ; adjust if your class expects extra args
        return ASLTranslator(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown --model '{name}'. Use one of: resnet50_bilstm, cnn_lstm")


# -----------------------------
# Training / Eval
# -----------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # -------- Train --------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_pbar.set_postfix({'loss': running_loss / max(total, 1), 'acc': 100. * correct / max(total, 1)})

        # -------- Val --------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_pbar.set_postfix({'loss': val_loss / max(val_total, 1), 'acc': 100. * val_correct / max(val_total, 1)})

        val_acc = 100. * val_correct / max(val_total, 1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')


# -----------------------------
# Main / CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ASL training with overfit/memorization harness")
    # Core
    p.add_argument("--data_dir", type=str, default="/content/my_project/DeepWS-main/asl_translator/src/data/processed")
    p.add_argument("--model", type=str, default="resnet50_bilstm", choices=["resnet50_bilstm", "cnn_lstm", "attention_resnet50"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_path", type=str, default="models/best_model.pth")
    p.add_argument("--seed", type=int, default=42)

    # Overfit harness
    p.add_argument("--overfit_n", type=int, default=0, help="If >0, train only on N samples (e.g., 30)")
    p.add_argument("--overfit_seed", type=int, default=1337, help="Seed for subset selection")
    p.add_argument("--val_on_train", action="store_true", help="Validate on the same tiny train subset")
    p.add_argument("--no_augment", action="store_true", help="Disable all stochastic data augmentation")
    p.add_argument("--dropout_override", type=float, default=None,
                   help="If set, force all Dropout layers to this value (e.g., 0.0)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Args: {args}")

    # Device & seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    set_seed(args.seed)

    # Transforms
    # NOTE: original code used a single (validation-like) transform for both train/val.
    # We keep that default; if you later add real augmentation, gate it with --no_augment.
    base_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.no_augment:
        train_transform = base_transform
    else:
        # Placeholder for future augmentation: currently identical to base to avoid repo behavior changes.
        train_transform = base_transform
    val_transform = base_transform

    # Datasets (single dataset like original; random_split uses same transform for both)
    dataset = ASLDataset(args.data_dir, transform=train_transform)

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Overfit harness: balanced 30-sample subset (or N)
    if args.overfit_n > 0:
        set_seed(args.overfit_seed)
        idxs = pick_balanced_subset(train_dataset, args.overfit_n, seed=args.overfit_seed)
        train_dataset = Subset(train_dataset, idxs)
        if args.val_on_train:
            val_dataset = train_dataset  # validate on the same tiny set

    # Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn_padd, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn_padd, pin_memory=True
    )

    # Model
    num_classes = len(dataset.classes)
    model = build_model(args.model, num_classes=num_classes).to(device)

    # Optionally turn off dropout for overfit check
    if args.dropout_override is not None:
        apply_dropout_override(model, args.dropout_override)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # If overfitting, strip weight decay to zero
    if args.overfit_n > 0:
        for g in optimizer.param_groups:
            g["weight_decay"] = 0.0

    # Train
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.epochs,
                device=device, save_path=args.save_path)


if __name__ == '__main__':
    main()
