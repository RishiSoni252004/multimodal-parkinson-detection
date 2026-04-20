"""
dataset.py — Unified Dataset Loader for Parkinson's Detection

Supports two input modes:
  1. CSV mode: Loads UCI parkinsons.data (195 samples, 22 features, binary label 'status')
  2. Audio mode: Loads WAV files from dataset/healthy/ and dataset/parkinson/ directories,
     extracts MFCC/Mel features, and returns them as tensors.

Features:
  - Auto-detects input type (CSV vs audio directory)
  - Cleans data (drops nulls, fixes types, removes non-feature columns)
  - Balances classes using RandomOverSampler
  - Splits into train/val/test (70/15/15)
  - Returns PyTorch DataLoader objects
  - Prints class distribution before and after balancing
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib
import warnings

warnings.filterwarnings("ignore")

# Try importing audio processing libraries
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("WARNING: librosa not installed. Audio mode will not work.")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_CSV_PATH = "dataset/parkinsons.data"
DEFAULT_AUDIO_DIR = "dataset/"
AUDIO_HEALTHY_DIR = "healthy"
AUDIO_PARKINSON_DIR = "parkinson"
TARGET_COLUMN = "status"
DROP_COLUMNS = ["name"]  # Non-feature columns to remove

# Audio processing defaults
AUDIO_SR = 16000
AUDIO_DURATION = 3.0  # seconds
AUDIO_MAX_SAMPLES = int(AUDIO_SR * AUDIO_DURATION)

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# DataLoader defaults
DEFAULT_BATCH_SIZE = 4  # M2 8GB memory safe


# ==============================================================================
# CSV DATASET LOADING
# ==============================================================================

def load_csv_dataset(csv_path: str = DEFAULT_CSV_PATH) -> tuple:
    """
    Load and clean the UCI Parkinson's CSV dataset.

    Args:
        csv_path: Path to the parkinsons.data CSV file.

    Returns:
        X (np.ndarray): Feature matrix (n_samples, n_features)
        y (np.ndarray): Label array (n_samples,)
        feature_names (list): List of feature column names
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at '{csv_path}'.")

    print(f"[CSV Mode] Loading dataset from: {csv_path}")

    # Read CSV — parkinsons.data is comma-separated with header
    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape}")

    # Drop non-feature columns
    for col in DROP_COLUMNS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            print(f"  Dropped column: '{col}'")

    # Drop duplicate rows
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Removed {n_dropped} duplicate rows.")

    # Drop rows with any null values
    n_before = len(df)
    df.dropna(inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Removed {n_dropped} rows with null values.")

    # Fix types — ensure all feature columns are numeric
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found. Columns: {list(df.columns)}")

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values.astype(np.int64)

    # Coerce feature columns to float
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Drop any columns that became all-NaN after coercion
    X.dropna(axis=1, how="all", inplace=True)
    # Fill remaining NaN with column median
    X.fillna(X.median(), inplace=True)

    feature_names = list(X.columns)
    X = X.values.astype(np.float32)

    print(f"  Cleaned shape: X={X.shape}, y={y.shape}")
    print(f"  Features: {len(feature_names)}")

    return X, y, feature_names


# ==============================================================================
# AUDIO DATASET LOADING
# ==============================================================================

def preprocess_single_audio(file_path: str) -> np.ndarray:
    """
    Load and preprocess a single audio file.

    Steps:
      1. Load with librosa at 16kHz mono
      2. Trim silence
      3. Normalize amplitude to [-1, 1]
      4. Pad/truncate to exactly AUDIO_DURATION seconds

    Args:
        file_path: Path to WAV/MP3 audio file.

    Returns:
        np.ndarray of shape (AUDIO_MAX_SAMPLES,) or None if failed.
    """
    if not HAS_LIBROSA:
        raise ImportError("librosa is required for audio processing.")

    try:
        y, sr = librosa.load(file_path, sr=AUDIO_SR)

        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) > 0:
            y = y_trimmed

        # Normalize amplitude to [-1, 1]
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        # Pad or truncate to fixed length
        if len(y) > AUDIO_MAX_SAMPLES:
            y = y[:AUDIO_MAX_SAMPLES]
        elif len(y) < AUDIO_MAX_SAMPLES:
            y = np.pad(y, (0, AUDIO_MAX_SAMPLES - len(y)), mode="constant")

        return y

    except Exception as e:
        print(f"  WARNING: Failed to process {file_path}: {e}")
        return None


def extract_audio_features(y: np.ndarray) -> np.ndarray:
    """
    Extract MFCC + Delta + Delta2 + Mel Spectrogram features from preprocessed audio.

    Args:
        y: Preprocessed audio array of shape (AUDIO_MAX_SAMPLES,)

    Returns:
        np.ndarray: 1D feature vector (248 features by default).
    """
    # MFCC (40 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=AUDIO_SR, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Delta MFCC
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mfcc_mean = np.mean(delta_mfcc.T, axis=0)

    # Delta-Delta MFCC
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta2_mfcc_mean = np.mean(delta2_mfcc.T, axis=0)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=AUDIO_SR)
    mel_mean = np.mean(mel.T, axis=0)

    # Concatenate all features into single vector
    feature_vector = np.concatenate([mfcc_mean, delta_mfcc_mean, delta2_mfcc_mean, mel_mean])
    return feature_vector.astype(np.float32)


def load_audio_dataset(audio_dir: str = DEFAULT_AUDIO_DIR) -> tuple:
    """
    Load audio files from directory structure and extract features.

    Expected structure:
      audio_dir/healthy/*.wav
      audio_dir/parkinson/*.wav

    Args:
        audio_dir: Root directory containing healthy/ and parkinson/ subdirectories.

    Returns:
        X (np.ndarray): Feature matrix (n_samples, n_features)
        y (np.ndarray): Label array (n_samples,)
        feature_names (list): Generic feature names
    """
    healthy_dir = os.path.join(audio_dir, AUDIO_HEALTHY_DIR)
    parkinson_dir = os.path.join(audio_dir, AUDIO_PARKINSON_DIR)

    if not os.path.exists(healthy_dir) and not os.path.exists(parkinson_dir):
        raise FileNotFoundError(
            f"Audio directories not found. Expected:\n"
            f"  {healthy_dir}\n"
            f"  {parkinson_dir}"
        )

    X_list = []
    y_list = []
    skipped = 0

    # Process healthy samples (label=0)
    if os.path.exists(healthy_dir):
        audio_files = [f for f in os.listdir(healthy_dir)
                       if f.lower().endswith((".wav", ".mp3"))
                       and "_converted" not in f.lower()]
        print(f"[Audio Mode] Processing {len(audio_files)} healthy audio files...")
        for fname in sorted(audio_files):
            fpath = os.path.join(healthy_dir, fname)
            audio = preprocess_single_audio(fpath)
            if audio is not None:
                features = extract_audio_features(audio)
                X_list.append(features)
                y_list.append(0)
            else:
                skipped += 1

    # Process parkinson samples (label=1)
    if os.path.exists(parkinson_dir):
        audio_files = [f for f in os.listdir(parkinson_dir)
                       if f.lower().endswith((".wav", ".mp3"))
                       and "_converted" not in f.lower()]
        print(f"[Audio Mode] Processing {len(audio_files)} parkinson audio files...")
        for fname in sorted(audio_files):
            fpath = os.path.join(parkinson_dir, fname)
            audio = preprocess_single_audio(fpath)
            if audio is not None:
                features = extract_audio_features(audio)
                X_list.append(features)
                y_list.append(1)
            else:
                skipped += 1

    if len(X_list) == 0:
        raise ValueError("No audio files could be loaded. Check your data directory.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Generate feature names
    n_features = X.shape[1]
    feature_names = [f"audio_feat_{i}" for i in range(n_features)]

    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    if skipped > 0:
        print(f"  Skipped {skipped} corrupted/unreadable files")

    return X, y, feature_names


# ==============================================================================
# CLASS BALANCING (Oversampling)
# ==============================================================================

def balance_classes(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Balance classes using random oversampling of the minority class.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)

    Returns:
        X_balanced (np.ndarray): Balanced feature matrix
        y_balanced (np.ndarray): Balanced label array
    """
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))

    print(f"\n  Class distribution BEFORE balancing:")
    for cls, count in class_dist.items():
        label = "Healthy" if cls == 0 else "Parkinson"
        pct = count / len(y) * 100
        print(f"    {label} (class {cls}): {count} samples ({pct:.1f}%)")

    # Find majority class count
    max_count = max(counts)

    X_balanced_parts = []
    y_balanced_parts = []

    for cls in unique:
        mask = y == cls
        X_cls = X[mask]
        y_cls = y[mask]

        if len(X_cls) < max_count:
            # Oversample minority class
            X_resampled, y_resampled = resample(
                X_cls, y_cls,
                replace=True,
                n_samples=max_count,
                random_state=42
            )
            X_balanced_parts.append(X_resampled)
            y_balanced_parts.append(y_resampled)
        else:
            X_balanced_parts.append(X_cls)
            y_balanced_parts.append(y_cls)

    X_balanced = np.concatenate(X_balanced_parts, axis=0)
    y_balanced = np.concatenate(y_balanced_parts, axis=0)

    # Shuffle the balanced dataset
    shuffle_idx = np.random.RandomState(42).permutation(len(X_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]

    unique_b, counts_b = np.unique(y_balanced, return_counts=True)
    class_dist_b = dict(zip(unique_b, counts_b))

    print(f"\n  Class distribution AFTER balancing:")
    for cls, count in class_dist_b.items():
        label = "Healthy" if cls == 0 else "Parkinson"
        pct = count / len(y_balanced) * 100
        print(f"    {label} (class {cls}): {count} samples ({pct:.1f}%)")

    return X_balanced, y_balanced


# ==============================================================================
# TRAIN / VAL / TEST SPLIT
# ==============================================================================

def split_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Split data into train/val/test sets (70/15/15).

    Uses stratified splitting to maintain class proportions.

    Args:
        X: Feature matrix
        y: Label array

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=42,
        stratify=y
    )

    # Second split: split the 30% temp into 50/50 → 15% val, 15% test
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        random_state=42,
        stratify=y_temp
    )

    print(f"\n  Split sizes:")
    print(f"    Train: {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.0f}%)")
    print(f"    Val:   {X_val.shape[0]} samples ({X_val.shape[0] / len(X) * 100:.0f}%)")
    print(f"    Test:  {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==============================================================================
# PYTORCH DATALOADER CREATION
# ==============================================================================

def create_dataloaders(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    scale: bool = True,
    save_scaler_path: str = None
) -> tuple:
    """
    Create PyTorch DataLoaders from numpy arrays.

    Args:
        X_train, X_val, X_test: Feature matrices
        y_train, y_val, y_test: Label arrays
        batch_size: Batch size for DataLoaders
        scale: Whether to apply StandardScaler normalization
        save_scaler_path: If provided, save the fitted StandardScaler to this path

    Returns:
        (train_loader, val_loader, test_loader, scaler_or_None)
    """
    scaler = None

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        if save_scaler_path:
            os.makedirs(os.path.dirname(save_scaler_path) or ".", exist_ok=True)
            joblib.dump(scaler, save_scaler_path)
            print(f"\n  Scaler saved to: {save_scaler_path}")

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size, shuffle=False
    )

    print(f"\n  DataLoaders created (batch_size={batch_size}):")
    print(f"    Train: {len(train_loader)} batches")
    print(f"    Val:   {len(val_loader)} batches")
    print(f"    Test:  {len(test_loader)} batches")

    return train_loader, val_loader, test_loader, scaler


# ==============================================================================
# MAIN AUTO-DETECT FUNCTION
# ==============================================================================

def load_dataset(
    csv_path: str = DEFAULT_CSV_PATH,
    audio_dir: str = DEFAULT_AUDIO_DIR,
    mode: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
    balance: bool = True,
    scale: bool = True,
    save_scaler_path: str = "models/scaler.pkl"
) -> dict:
    """
    Main entry point. Auto-detects whether to use CSV or audio loading.

    Args:
        csv_path: Path to CSV dataset file.
        audio_dir: Path to audio directory containing healthy/ and parkinson/ subdirs.
        mode: 'csv', 'audio', or 'auto' (auto-detect).
        batch_size: Batch size for DataLoaders.
        balance: Whether to balance classes via oversampling.
        scale: Whether to StandardScale features.
        save_scaler_path: Path to save the fitted scaler.

    Returns:
        dict with keys:
          - 'train_loader': PyTorch DataLoader for training
          - 'val_loader':   PyTorch DataLoader for validation
          - 'test_loader':  PyTorch DataLoader for testing
          - 'scaler':       Fitted StandardScaler (or None)
          - 'feature_names': List of feature names
          - 'input_size':   Number of features (int)
          - 'mode':         'csv' or 'audio'
    """
    print("=" * 60)
    print(" DATASET LOADER — Parkinson's Disease Detection")
    print("=" * 60)

    # Auto-detect mode
    if mode == "auto":
        has_csv = os.path.exists(csv_path)
        has_audio = (
            os.path.exists(os.path.join(audio_dir, AUDIO_HEALTHY_DIR)) or
            os.path.exists(os.path.join(audio_dir, AUDIO_PARKINSON_DIR))
        )

        if has_csv:
            mode = "csv"
            print(f"\n  Auto-detected: CSV mode (found {csv_path})")
        elif has_audio:
            mode = "audio"
            print(f"\n  Auto-detected: Audio mode (found audio dirs in {audio_dir})")
        else:
            raise FileNotFoundError(
                f"No dataset found. Provide either:\n"
                f"  - CSV file at: {csv_path}\n"
                f"  - Audio dirs at: {audio_dir}/healthy/ and {audio_dir}/parkinson/"
            )

    # Load data
    if mode == "csv":
        X, y, feature_names = load_csv_dataset(csv_path)
    elif mode == "audio":
        X, y, feature_names = load_audio_dataset(audio_dir)
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Use 'csv', 'audio', or 'auto'.")

    # Balance classes
    if balance:
        X, y = balance_classes(X, y)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Create DataLoaders
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        batch_size=batch_size,
        scale=scale,
        save_scaler_path=save_scaler_path
    )

    # Save feature names
    os.makedirs("models", exist_ok=True)
    joblib.dump(feature_names, "models/selected_features.pkl")

    input_size = X_train.shape[1]

    print(f"\n  Input feature size: {input_size}")
    print("=" * 60)
    print(" DATASET LOADING COMPLETE")
    print("=" * 60)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "feature_names": feature_names,
        "input_size": input_size,
        "mode": mode,
    }


# ==============================================================================
# SELF-TEST
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" RUNNING DATASET SELF-TEST")
    print("=" * 60)

    # Test 1: CSV mode
    print("\n--- Test 1: CSV Mode ---")
    try:
        result = load_dataset(
            csv_path="dataset/parkinsons.data",
            mode="csv",
            batch_size=4,
            balance=True,
            scale=True,
            save_scaler_path="models/scaler.pkl"
        )
        print(f"\n  ✅ CSV mode OK")
        print(f"     Input size: {result['input_size']}")
        print(f"     Train batches: {len(result['train_loader'])}")
        print(f"     Val batches: {len(result['val_loader'])}")
        print(f"     Test batches: {len(result['test_loader'])}")

        # Verify a batch
        for X_batch, y_batch in result["train_loader"]:
            print(f"     Sample batch: X={X_batch.shape}, y={y_batch.shape}")
            break

    except Exception as e:
        print(f"  ❌ CSV mode FAILED: {e}")

    # Test 2: Audio mode
    print("\n--- Test 2: Audio Mode ---")
    try:
        result_audio = load_dataset(
            audio_dir="dataset/",
            mode="audio",
            batch_size=4,
            balance=True,
            scale=True,
            save_scaler_path="models/voice_dl_scaler.pkl"
        )
        print(f"\n  ✅ Audio mode OK")
        print(f"     Input size: {result_audio['input_size']}")
        print(f"     Train batches: {len(result_audio['train_loader'])}")

        for X_batch, y_batch in result_audio["train_loader"]:
            print(f"     Sample batch: X={X_batch.shape}, y={y_batch.shape}")
            break

    except Exception as e:
        print(f"  ❌ Audio mode FAILED: {e}")

    # Test 3: Auto mode
    print("\n--- Test 3: Auto-detect Mode ---")
    try:
        result_auto = load_dataset(mode="auto", batch_size=4)
        print(f"\n  ✅ Auto mode selected: {result_auto['mode']}")
    except Exception as e:
        print(f"  ❌ Auto mode FAILED: {e}")

    print("\n" + "=" * 60)
    print(" SELF-TEST COMPLETE")
    print("=" * 60)
