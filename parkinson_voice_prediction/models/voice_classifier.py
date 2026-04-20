"""
voice_classifier.py — Optimized PyTorch Classifier on Wav2Vec 2.0 Embeddings

Architecture:
  Linear(768, 256) → ReLU → Dropout(0.3/0.5) → Linear(256, 2)

Performance optimizations (Step 7):
  - Gradient checkpointing awareness
  - torch.no_grad() on all inference
  - tqdm progress bars
  - Auto dropout/weight_decay adjustment on overfitting
  - Memory-safe for M2 8GB
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ==============================================================================
# DEVICE
# ==============================================================================

def get_device() -> torch.device:
    """Get best available device: MPS > CPU. Never CUDA."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==============================================================================
# CLASSIFIER MODEL
# ==============================================================================

class VoiceClassifier(nn.Module):
    """
    Small classifier head for Wav2Vec2 embeddings.
    Dropout is configurable (0.3 default, 0.5 if overfitting detected).
    """

    def __init__(self, input_dim: int = 768, num_classes: int = 2, dropout: float = 0.3):
        super(VoiceClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: [batch, 768] → Output: [batch, 2]"""
        return self.classifier(x)


# ==============================================================================
# EMBEDDING EXTRACTION
# ==============================================================================

def extract_all_embeddings(data_dir: str = "dataset/") -> tuple:
    """
    Extract Wav2Vec2 embeddings for all audio files in the dataset.

    Args:
        data_dir: Root directory containing healthy/ and parkinson/ subdirs.

    Returns:
        (embeddings, labels): numpy arrays of shape (n, 768) and (n,)
    """
    from models.wav2vec_model import extract_embeddings

    healthy_dir = os.path.join(data_dir, "healthy")
    parkinson_dir = os.path.join(data_dir, "parkinson")

    embeddings = []
    labels = []
    skipped = 0

    for label, class_dir, class_name in [(0, healthy_dir, "healthy"), (1, parkinson_dir, "parkinson")]:
        if not os.path.exists(class_dir):
            continue
        files = [f for f in sorted(os.listdir(class_dir))
                 if f.lower().endswith((".wav", ".mp3"))
                 and "_converted" not in f.lower()]

        desc = f"  Extracting {class_name}"
        iterator = tqdm(files, desc=desc, unit="file") if HAS_TQDM else files
        if not HAS_TQDM:
            print(f"{desc} ({len(files)} files)...")

        for fname in iterator:
            fpath = os.path.join(class_dir, fname)
            try:
                emb = extract_embeddings(fpath)
                embeddings.append(emb.squeeze(0).numpy())
                labels.append(label)
            except Exception as e:
                skipped += 1

    if len(embeddings) == 0:
        raise ValueError("No embeddings could be extracted.")

    X = np.array(embeddings, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    print(f"\n  Total: {len(X)} embeddings ({skipped} skipped)")
    print(f"  Classes: healthy={np.sum(y == 0)}, parkinson={np.sum(y == 1)}")
    return X, y


# ==============================================================================
# TRAINING LOOP WITH PERFORMANCE OPTIMIZATIONS
# ==============================================================================

def train_voice_classifier(
    data_dir: str = "dataset/",
    max_epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    patience: int = 5,
    dropout: float = 0.3,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
):
    """
    Full training pipeline with performance optimizations.

    Optimizations:
      - torch.no_grad() for all validation/test inference
      - tqdm progress bars
      - Auto-detects overfitting: if val_loss rises 3 consecutive epochs while
        train_loss drops, increases dropout to 0.5 and weight_decay to 1e-3
      - Batch size 4 for M2 8GB safety
      - gradient checkpointing ready (not needed for this small classifier)
    """
    print("=" * 60)
    print(" VOICE CLASSIFIER TRAINING (Optimized)")
    print("=" * 60)

    device = get_device()
    print(f"\n  Device: {device}")

    # Step 1: Extract embeddings
    print("\n--- Step 1: Extracting Wav2Vec2 Embeddings ---")
    X, y = extract_all_embeddings(data_dir)

    # Step 2: Split 70/15/15
    print("\n--- Step 2: Splitting Data ---")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Class weights
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weights = torch.tensor(
        [total / (len(unique) * c) for c in counts], dtype=torch.float32
    ).to(device)
    print(f"  Class weights: {class_weights.tolist()}")

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                       torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                       torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                       torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )

    # Model
    print("\n--- Step 3: Training ---")
    model = VoiceClassifier(input_dim=768, num_classes=2, dropout=dropout)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    patience_counter = 0
    overfit_counter = 0
    prev_train_loss = float("inf")
    prev_val_loss = float("inf")
    overfitting_adjusted = False

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "voice_best.pt")

    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8} | {'Status':>10}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_batches = 0

        for inputs, labels_batch in train_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)

        # --- Validate (with torch.no_grad) ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels_batch in val_loader:
                inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                val_batches += 1
                _, predicted = torch.max(outputs, 1)
                val_total += labels_batch.size(0)
                val_correct += (predicted == labels_batch).sum().item()

        avg_val_loss = val_loss / max(val_batches, 1)
        val_acc = val_correct / max(val_total, 1) * 100

        # --- Overfitting detection ---
        status = ""
        if avg_train_loss < prev_train_loss and avg_val_loss > prev_val_loss:
            overfit_counter += 1
            if overfit_counter >= 3 and not overfitting_adjusted:
                # Increase regularization
                dropout = 0.5
                weight_decay = 1e-3
                model = VoiceClassifier(input_dim=768, num_classes=2, dropout=0.5).to(device)
                # Reload best weights if available
                if os.path.exists(checkpoint_path):
                    cp = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    model.load_state_dict(cp["model_state_dict"])
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
                overfitting_adjusted = True
                overfit_counter = 0
                status = "↑DropOut"
                print(f"  ** Overfitting detected: dropout→0.5, weight_decay→1e-3 **")
        else:
            overfit_counter = 0

        prev_train_loss = avg_train_loss
        prev_val_loss = avg_val_loss

        # --- Early stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": 768, "num_classes": 2, "dropout": dropout,
                "epoch": epoch + 1, "val_loss": best_val_loss, "val_acc": val_acc,
            }, checkpoint_path)
            status = "★ saved"
        else:
            patience_counter += 1
            if patience_counter >= patience:
                status = "STOP"
                print(f"  {epoch+1:>5} | {avg_train_loss:>10.4f} | {avg_val_loss:>10.4f} | {val_acc:>7.2f}% | {status:>10}")
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break

        print(f"  {epoch+1:>5} | {avg_train_loss:>10.4f} | {avg_val_loss:>10.4f} | {val_acc:>7.2f}% | {status:>10}")

    print(f"\n  Best checkpoint: {checkpoint_path} (val_loss={best_val_loss:.4f})")

    # --- Final evaluation on test set ---
    print("\n--- Step 4: Final Evaluation ---")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VoiceClassifier(
        input_dim=checkpoint.get("input_dim", 768),
        num_classes=checkpoint.get("num_classes", 2),
        dropout=checkpoint.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels_batch in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f} (macro)")
    print(f"  Recall:    {rec:.4f} (macro)")
    print(f"  F1 Score:  {f1:.4f} (macro)")
    print(f"\n  Confusion Matrix:")
    print(f"               Healthy  Parkinson")
    print(f"  Healthy       {cm[0][0]:>4}     {cm[0][1]:>4}")
    print(f"  Parkinson     {cm[1][0]:>4}     {cm[1][1]:>4}")

    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE")
    print("=" * 60)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "confusion_matrix": cm, "checkpoint_path": checkpoint_path}


# ==============================================================================
# INFERENCE
# ==============================================================================

def predict_with_classifier(
    audio_path: str,
    checkpoint_path: str = "checkpoints/voice_best.pt",
) -> tuple:
    """
    Predict Parkinson's from audio using trained classifier.

    Args:
        audio_path: Path to audio file.
        checkpoint_path: Path to model checkpoint.

    Returns:
        (prediction_label, confidence): label is str, confidence is float.
    """
    from models.wav2vec_model import extract_embeddings

    device = get_device()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VoiceClassifier(
        input_dim=checkpoint.get("input_dim", 768),
        num_classes=checkpoint.get("num_classes", 2),
        dropout=checkpoint.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    embedding = extract_embeddings(audio_path).to(device)

    with torch.no_grad():
        logits = model(embedding)
        probs = torch.softmax(logits, dim=1)
        prob_parkinson = probs[0][1].item()
        prediction = 1 if prob_parkinson >= 0.5 else 0

    label = "Parkinson Detected" if prediction == 1 else "Healthy"
    confidence = prob_parkinson if prediction == 1 else (1 - prob_parkinson)
    return label, confidence


if __name__ == "__main__":
    train_voice_classifier(data_dir="dataset/", max_epochs=30, batch_size=4,
                           lr=1e-4, patience=5, dropout=0.3, checkpoint_dir="checkpoints")
