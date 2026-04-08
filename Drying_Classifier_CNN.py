import os
import gc
import argparse
import multiprocessing
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

# Plot: Training Curves
def plot_metrics(history, save_path=None):
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(12, 4))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs, history['val_acc'],   label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'],   label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Training curves -> {save_path}")

    plt.show()
    plt.close()

# Plot: Confusion Matrix
def plot_confusion_matrix(cm_counts, cm_perc, class_names, title, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm_perc, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    thresh = cm_perc.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i,
                    f"{cm_counts[i, j]} ({cm_perc[i, j]:.2f}%)",
                    ha="center", va="center",
                    color="white" if cm_perc[i, j] > thresh else "black")

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] {title} -> {save_path}")

    plt.show()
    plt.close()

# Compute Confusion Matrix
def compute_confusion_matrix(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm_counts = confusion_matrix(all_labels, all_preds)
    cm_perc = cm_counts.astype('float') / cm_counts.sum(axis=1)[:, np.newaxis] * 100
    cm_perc = np.round(cm_perc, 2)
    return cm_counts, cm_perc

# EarlyStopping (Based on val_acc & EMA val_loss)
class EarlyStopping:
    def __init__(self, patience=40, min_delta=1e-4, verbose=False):
        self.patience   = patience
        self.min_delta  = min_delta
        self.verbose    = verbose

        self.best_acc   = None
        self.best_loss  = None
        self.counter    = 0

    def step(self, val_acc, val_loss):
        # First epoch initialization
        if self.best_acc is None:
            self.best_acc  = val_acc
            self.best_loss = val_loss
            return False

        acc_improved  = val_acc  > self.best_acc  + self.min_delta
        loss_improved = val_loss < self.best_loss - self.min_delta

        if acc_improved or loss_improved:
            self.best_acc  = max(self.best_acc,  val_acc)
            self.best_loss = min(self.best_loss, val_loss)
            self.counter   = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStop] counter = {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            if self.verbose:
                print("[EarlyStop] Triggered training stop.")
            return True

        return False

# Main Execution
def main():
    # Setup Argument Parser for flexible execution
    parser = argparse.ArgumentParser(description="Train lightweight CNN for drying stage classification")
    parser.add_argument('--train_dir', type=str, default='./dataset/train', help='Path to training data')
    parser.add_argument('--val_dir', type=str, default='./dataset/validation', help='Path to validation data')
    parser.add_argument('--test_dir', type=str, default='./dataset/test', help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model and plots')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    img_h, img_w = 115, 40
    batch_size = 64
    epochs     = 300

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using {device}")

    ckpt = os.path.join(args.output_dir, 'best_model.pth')
    curve_path = os.path.join(args.output_dir, 'training_curve.png')

    # Transform (Grayscale & Data Augmentation)  
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),                    
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
    ])

    # Dataset & Loader
    train_ds = datasets.ImageFolder(args.train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(args.val_dir,   transform=val_tf)
    test_ds  = datasets.ImageFolder(args.test_dir,  transform=val_tf)
    class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model Architecture
    class MyCNN(nn.Module):
        def __init__(self, n_cls):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                # Block 2
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                # Block 3
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
                # Block 4
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
                # Block 5
                nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
                # Block 6
                nn.Conv2d(256, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            )

            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout     = nn.Dropout(0.5)
            self.classifier  = nn.Linear(64, n_cls)

        def forward(self, x):
            x = self.features(x)
            x = self.global_pool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            return self.classifier(x)

    model = MyCNN(len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=10,
        threshold=1e-3,
        cooldown=5,
        min_lr=1e-6
    )

    # EarlyStopping Trigger
    early_stopper = EarlyStopping(patience=40, verbose=True)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    ema_val_loss = None
    ema_alpha = 0.9

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_correct += (out.argmax(1) == y).sum().item()

        train_loss /= len(train_ds)
        train_acc  = train_correct / len(train_ds)

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)

                val_loss += loss.item() * X.size(0)
                val_correct += (out.argmax(1) == y).sum().item()

        val_loss /= len(val_ds)
        val_acc  = val_correct / len(val_ds)

        # EMA smoothing for validation loss
        if ema_val_loss is None:
            ema_val_loss = val_loss
        else:
            ema_val_loss = ema_alpha * ema_val_loss + (1 - ema_alpha) * val_loss

        # Save History
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"[{epoch}/{epochs}] "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val Loss(raw)={val_loss:.4f} EMA={ema_val_loss:.4f} "
              f"Acc={val_acc:.4f}")

        # Update Scheduler
        scheduler.step(ema_val_loss)

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt)
            print("→ Best Model Saved")

        # Check Early Stopping
        if early_stopper.step(val_acc, ema_val_loss):
            break

    # Save & Visualize Training Curves
    plot_metrics(history, save_path=curve_path)

    # Test Evaluation
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    test_loss = 0.0
    test_correct = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            test_loss += loss.item() * X.size(0)
            test_correct += (out.argmax(1) == y).sum().item()

    test_loss /= len(test_ds)
    test_acc = test_correct / len(test_ds)
    print(f"[Test Evaluation] Loss={test_loss:.4f} Acc={test_acc:.4f}")

    # Generate & Save Confusion Matrices
    for name, loader in [('Train', train_loader),
                         ('Validation', val_loader),
                         ('Test', test_loader)]:
        cm_counts, cm_perc = compute_confusion_matrix(loader, model, device)
        cm_path = os.path.join(
            args.output_dir,
            f'confusion_matrix_{name.lower()}.png'
        )
        plot_confusion_matrix(
            cm_counts, cm_perc, class_names,
            title=f'{name} Confusion Matrix (Count & %)',
            save_path=cm_path
        )

    # Clear GPU memory
    del model, optimizer, scheduler, criterion
    del train_loader, val_loader, test_loader
    del train_ds, val_ds, test_ds

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[GPU] torch.cuda.empty_cache() Complete")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()