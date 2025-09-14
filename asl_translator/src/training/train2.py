import os
import sys
import argparse
import random
import logging
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_lstm import ASLTranslator, ASLDataLoader
from models.resnet50_bilstm import ResNet50BiLSTM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG = {
    'IMG_SIZE': 224,
    'IMAGENET_MEAN': [0.485, 0.456, 0.406],
    'IMAGENET_STD': [0.229, 0.224, 0.225],
    'MAX_FRAMES': 64,  # Limit video length
    'MIN_FRAMES': 8,   # Minimum required frames
    'TARGET_FRAMES': 32,  # Target number of frames to sample
}


# -----------------------------
# Utility Classes
# -----------------------------
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def __call__(self, val_acc: float) -> bool:
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class MetricsTracker:
    """Track training metrics."""
    
    def __init__(self):
        self.history = []
    
    def update(self, epoch: int, train_loss: float, train_acc: float, 
               val_loss: float, val_acc: float, lr: float):
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr
        }
        self.history.append(metrics)
        return metrics


# -----------------------------
# Dataset
# -----------------------------
class ASLDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, target_frames: int = CONFIG['TARGET_FRAMES']):
        self.data_dir = data_dir
        self.transform = transform
        self.target_frames = target_frames
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        self.classes = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        self.targets = []
        
        logger.info(f"Loading dataset from {data_dir}")
        self._load_samples()
        logger.info(f"Loaded {len(self.samples)} valid video samples across {len(self.classes)} classes")

    def _load_samples(self):
        """Load and validate video samples."""
        for class_name in tqdm(self.classes, desc="Loading classes"):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            valid_count = 0
            for video_file in os.listdir(class_dir):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(class_dir, video_file)
                    if self._validate_video_file(video_path):
                        self.samples.append((video_path, class_name))
                        self.targets.append(self.class_to_idx[class_name])
                        valid_count += 1
            
            logger.info(f"Class '{class_name}': {valid_count} valid videos")

    def _validate_video_file(self, video_path: str) -> bool:
        """Check if video file is readable and has minimum frames."""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return frame_count >= CONFIG['MIN_FRAMES']
        except Exception as e:
            logger.warning(f"Invalid video file {video_path}: {e}")
            return False

    def _sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Sample fixed number of frames from video."""
        num_frames = frames.size(0)
        
        if num_frames <= self.target_frames:
            return frames
        
        # Sample frames uniformly across the video
        indices = np.linspace(0, num_frames - 1, self.target_frames, dtype=int)
        return frames[indices]

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, class_name = self.samples[idx]
        
        try:
            loader = ASLDataLoader(video_path, self.transform)
            frames = loader.load_video()  # expected shape: [T, C, H, W]
            
            if frames is None or frames.size(0) == 0:
                raise ValueError(f"No frames loaded from {video_path}")
            
            # Sample frames to fixed length
            frames = self._sample_frames(frames)
            
            # Ensure minimum frames
            if frames.size(0) < CONFIG['MIN_FRAMES']:
                # Repeat frames if too few
                repeat_factor = int(np.ceil(CONFIG['MIN_FRAMES'] / frames.size(0)))
                frames = frames.repeat(repeat_factor, 1, 1, 1)[:CONFIG['MIN_FRAMES']]
            
            label = self.class_to_idx[class_name]
            return frames, label
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            # Return dummy data to prevent training crash
            dummy_frames = torch.zeros(CONFIG['MIN_FRAMES'], 3, CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])
            return dummy_frames, self.class_to_idx[class_name]


# -----------------------------
# Improved Collate Function
# -----------------------------
def collate_fn_padd(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function with sequence length tracking."""
    lengths = [item[0].size(0) for item in batch]
    max_len = max(lengths)
    
    padded_inputs, labels, seq_lengths = [], [], []
    for inputs, label in batch:
        seq_length = inputs.size(0)
        padding_size = max_len - seq_length
        
        if padding_size > 0:
            padded_input = F.pad(inputs, (0, 0, 0, 0, 0, 0, 0, padding_size))
        else:
            padded_input = inputs
            
        padded_inputs.append(padded_input)
        labels.append(label)
        seq_lengths.append(seq_length)
    
    return torch.stack(padded_inputs), torch.tensor(labels), torch.tensor(seq_lengths)


# -----------------------------
# Utils for overfit harness
# -----------------------------
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split(dataset: Dataset, train_ratio: float = 0.8, 
                    random_state: int = 42) -> Tuple[Subset, Subset]:
    """Create stratified train/validation split."""
    indices = np.arange(len(dataset))
    labels = [dataset.targets[i] for i in indices]
    
    train_idx, val_idx = train_test_split(
        indices, test_size=1-train_ratio, 
        stratify=labels, random_state=random_state
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def pick_balanced_subset(dataset: Dataset, n: int, seed: int = 1337) -> List[int]:
    """Choose up to n indices, round-robin over classes."""
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
    """Override all dropout rates in model."""
    def set_drop(m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.p = p
    model.apply(set_drop)


def build_model(name: str, num_classes: int) -> nn.Module:
    """Build model by name."""
    name = name.lower()
    if name == "resnet50_bilstm":
        return ResNet50BiLSTM(num_classes=num_classes)
    elif name == "cnn_lstm":
        return ASLTranslator(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown --model '{name}'. Use one of: resnet50_bilstm, cnn_lstm")


# -----------------------------
# Training / Eval
# -----------------------------
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               criterion: nn.Module, optimizer: torch.optim.Optimizer, 
               num_epochs: int, device: torch.device, save_path: str,
               use_amp: bool = True, patience: int = 10) -> Dict[str, Any]:
    """Train model with improved monitoring and early stopping."""
    
    best_val_acc = 0.0
    scaler = GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    early_stopping = EarlyStopping(patience=patience, min_delta=0.01)
    metrics_tracker = MetricsTracker()

    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    for epoch in range(num_epochs):
        # -------- Train --------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels, seq_lengths in train_pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            if scaler is not None:
                with autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_pbar.set_postfix({
                'loss': f"{running_loss / max(total, 1):.4f}", 
                'acc': f"{100. * correct / max(total, 1):.2f}%"
            })

        train_loss = running_loss / max(total, 1)
        train_acc = 100. * correct / max(total, 1)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels, seq_lengths in val_pbar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if scaler is not None:
                    with autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_pbar.set_postfix({
                    'loss': f"{val_loss / max(val_total, 1):.4f}", 
                    'acc': f"{100. * val_correct / max(val_total, 1):.2f}%"
                })

        val_loss = val_loss / max(val_total, 1)
        val_acc = 100. * val_correct / max(val_total, 1)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Track metrics
        current_lr = optimizer.param_groups[0]['lr']
        metrics = metrics_tracker.update(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                   f"LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'metrics_history': metrics_tracker.history
            }, save_path)
            logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        # Early stopping check
        if early_stopping(val_acc):
            logger.info(f'Early stopping triggered at epoch {epoch+1}')
            break

    return {
        'best_val_acc': best_val_acc,
        'metrics_history': metrics_tracker.history
    }


# -----------------------------
# Main / CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ASL training with overfit/memorization harness")
    # Core
    p.add_argument("--data_dir", type=str, default="/content/my_project/DeepWS-main/asl_translator/src/data/processed")
    p.add_argument("--model", type=str, default="resnet50_bilstm", choices=["resnet50_bilstm", "cnn_lstm"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_path", type=str, default="models/best_model.pth")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_frames", type=int, default=32, help="Number of frames to sample from each video")

    # Training enhancements
    p.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")

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
    logger.info(f"Arguments: {args}")

    # Device & seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    if device.type == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name()}')
        logger.info(f'GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f}GB')
    
    set_seed(args.seed)

    # Transforms
    base_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=CONFIG['IMAGENET_MEAN'], std=CONFIG['IMAGENET_STD'])
    ])

    if args.no_augment:
        train_transform = base_transform
    else:
        # Add augmentation in future
        train_transform = base_transform
    val_transform = base_transform

    # Dataset with frame sampling
    CONFIG['TARGET_FRAMES'] = args.target_frames
    dataset = ASLDataset(args.data_dir, transform=train_transform, target_frames=args.target_frames)

    # Stratified split for better class balance
    train_dataset, val_dataset = stratified_split(dataset, train_ratio=0.8, random_state=args.seed)

    # Overfit harness
    if args.overfit_n > 0:
        logger.info(f"Overfit mode: using {args.overfit_n} samples")
        set_seed(args.overfit_seed)
        idxs = pick_balanced_subset(train_dataset, args.overfit_n, seed=args.overfit_seed)
        train_dataset = Subset(train_dataset, idxs)
        if args.val_on_train:
            val_dataset = train_dataset
            logger.info("Validating on training set for overfit check")

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn_padd, 
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn_padd,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Model
    num_classes = len(dataset.classes)
    logger.info(f"Building {args.model} model with {num_classes} classes")
    model = build_model(args.model, num_classes=num_classes).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Optionally turn off dropout for overfit check
    if args.dropout_override is not None:
        apply_dropout_override(model, args.dropout_override)
        logger.info(f"Dropout override set to {args.dropout_override}")

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # If overfitting, strip weight decay
    if args.overfit_n > 0:
        for g in optimizer.param_groups:
            g["weight_decay"] = 0.0
        logger.info("Weight decay disabled for overfit mode")

    # Train
    use_amp = not args.no_amp and device.type == 'cuda'
    logger.info(f"Mixed precision training: {'Enabled' if use_amp else 'Disabled'}")
    
    results = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=args.epochs, device=device, save_path=args.save_path,
        use_amp=use_amp, patience=args.patience
    )
    
    logger.info(f"Training completed! Best validation accuracy: {results['best_val_acc']:.2f}%")


if __name__ == '__main__':
    main()
