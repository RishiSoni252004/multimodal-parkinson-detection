"""
wav2vec_model.py — Fixed Wav2Vec 2.0 Pipeline for Parkinson's Detection

Changes from original:
  - Uses facebook/wav2vec2-base (feature extractor) NOT wav2vec2-base-960h (ASR model)
  - Proper MPS/CPU device handling (NEVER CUDA)
  - Reusable extract_embeddings() function with correct tensor shapes
  - Mean-pooled embeddings: [batch, 768]
  - Full torch.no_grad() for all inference
  - Accepts both file paths and preprocessed tensors
"""

import os
import sys
import torch
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.filterwarnings("ignore")


# ==============================================================================
# DEVICE SELECTION — MPS on Apple Silicon, else CPU. NEVER CUDA.
# ==============================================================================

def get_device() -> torch.device:
    """
    Get the best available device for this system.
    Priority: MPS (Apple Silicon) > CPU. Never uses CUDA.

    Returns:
        torch.device: The selected device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==============================================================================
# EMBEDDING EXTRACTION
# ==============================================================================

# Module-level cache to avoid reloading the model on every call
_cached_processor = None
_cached_model = None
_cached_device = None


def _load_wav2vec_model(device: torch.device = None):
    """
    Load (or return cached) Wav2Vec2 processor and model.

    Uses facebook/wav2vec2-base — the base feature extractor model,
    NOT wav2vec2-base-960h which is fine-tuned for ASR.

    Args:
        device: Target device. If None, auto-detects.

    Returns:
        (processor, model, device)
    """
    global _cached_processor, _cached_model, _cached_device

    if device is None:
        device = get_device()

    if _cached_model is not None and _cached_device == device:
        return _cached_processor, _cached_model, _cached_device

    print(f"  Loading Wav2Vec2 model (facebook/wav2vec2-base) on {device}...")

    _cached_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    _cached_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    _cached_model.eval()

    # Move to MPS if available — Wav2Vec2 works on MPS with PyTorch 2.x
    # Note: if MPS causes issues with specific ops, we silently fall back to CPU
    try:
        _cached_model = _cached_model.to(device)
        _cached_device = device
    except Exception:
        print(f"  Warning: Failed to move model to {device}, falling back to CPU.")
        _cached_model = _cached_model.to(torch.device("cpu"))
        _cached_device = torch.device("cpu")

    return _cached_processor, _cached_model, _cached_device


def extract_embeddings(audio_input, device: torch.device = None) -> torch.Tensor:
    """
    Extract mean-pooled Wav2Vec2 embeddings from audio.

    Accepts EITHER:
      - A torch.Tensor of shape [1, N] or [N] (raw waveform at 16kHz)
      - A numpy array of shape (N,) (raw waveform at 16kHz)
      - A file path (string) — will be loaded and preprocessed via preprocess.py

    Args:
        audio_input: Audio waveform tensor, numpy array, or file path string.
        device: Target device. If None, auto-detects.

    Returns:
        torch.Tensor: Mean-pooled embeddings of shape [1, 768] (on CPU for downstream use).
    """
    processor, model, model_device = _load_wav2vec_model(device)

    # === Handle different input types ===

    if isinstance(audio_input, str):
        # File path — use preprocess.py
        from preprocess import preprocess_audio
        audio_tensor = preprocess_audio(audio_input)  # [1, 48000]
        waveform = audio_tensor.squeeze(0).numpy()  # (48000,)

    elif isinstance(audio_input, torch.Tensor):
        # Tensor input
        if audio_input.dim() == 2:
            waveform = audio_input.squeeze(0).cpu().numpy()  # [1, N] → (N,)
        elif audio_input.dim() == 1:
            waveform = audio_input.cpu().numpy()  # (N,)
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got {audio_input.dim()}D")

    elif isinstance(audio_input, np.ndarray):
        if audio_input.ndim == 2:
            waveform = audio_input.squeeze(0)  # [1, N] → (N,)
        elif audio_input.ndim == 1:
            waveform = audio_input
        else:
            raise ValueError(f"Expected 1D or 2D array, got {audio_input.ndim}D")

    else:
        raise TypeError(
            f"audio_input must be a file path (str), torch.Tensor, or numpy array. "
            f"Got: {type(audio_input)}"
        )

    # Ensure float32
    waveform = waveform.astype(np.float32)

    # === Process through Wav2Vec2 ===

    # Processor converts waveform to model-ready input_values
    inputs = processor(
        waveform,
        return_tensors="pt",
        sampling_rate=16000,
        padding=True
    )

    input_values = inputs.input_values.to(model_device)

    # Extract hidden states with no gradient computation
    with torch.no_grad():
        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state  # [1, seq_len, 768]

    # Mean-pool over the time/sequence dimension → [1, 768]
    embedding = torch.mean(hidden_states, dim=1)  # [1, 768]

    # Always return on CPU for downstream sklearn/numpy compatibility
    return embedding.cpu()


def extract_embeddings_batch(audio_list: list, device: torch.device = None) -> torch.Tensor:
    """
    Extract embeddings for a batch of audio inputs.

    Args:
        audio_list: List of file paths, tensors, or numpy arrays.
        device: Target device.

    Returns:
        torch.Tensor: Shape [batch_size, 768]
    """
    embeddings = []
    for audio in audio_list:
        try:
            emb = extract_embeddings(audio, device)  # [1, 768]
            embeddings.append(emb)
        except Exception as e:
            print(f"  Warning: Skipping an audio input: {e}")

    if not embeddings:
        raise ValueError("No embeddings could be extracted from the batch.")

    return torch.cat(embeddings, dim=0)  # [batch_size, 768]


# ==============================================================================
# PARKINSON MODEL (Wav2Vec + Classifier)
# ==============================================================================

class Wav2VecParkinsonModel:
    """
    Complete Parkinson's detection pipeline using Wav2Vec2 embeddings
    + sklearn MLPClassifier.
    """

    def __init__(self):
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            alpha=0.001,  # L2 regularization
        )
        self.classifier_path = "models/wav2vec_classifier.pkl"
        self._classifier_loaded = False

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract Wav2Vec2 embedding from an audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            np.ndarray: Embedding vector of shape (768,), or None on failure.
        """
        try:
            emb = extract_embeddings(audio_path)  # [1, 768]
            return emb.squeeze(0).numpy()  # (768,)
        except Exception as e:
            print(f"  Error extracting embedding from {audio_path}: {e}")
            return None

    def load_dataset_from_directory(self, root_dir: str) -> tuple:
        """
        Load audio files from directory and extract Wav2Vec2 embeddings.

        Expected structure:
          root_dir/healthy/*.wav
          root_dir/parkinson/*.wav

        Args:
            root_dir: Root directory path.

        Returns:
            (X, y): numpy arrays of embeddings and labels.
        """
        X, y = [], []

        # Process healthy samples (label=0)
        healthy_dir = os.path.join(root_dir, "healthy")
        if os.path.exists(healthy_dir):
            files = [f for f in sorted(os.listdir(healthy_dir))
                     if f.lower().endswith((".wav", ".mp3"))
                     and "_converted" not in f.lower()]
            print(f"  Processing {len(files)} healthy audio files...")
            for f in files:
                path = os.path.join(healthy_dir, f)
                emb = self.extract_embedding(path)
                if emb is not None:
                    X.append(emb)
                    y.append(0)

        # Process parkinson samples (label=1)
        parkinson_dir = os.path.join(root_dir, "parkinson")
        if os.path.exists(parkinson_dir):
            files = [f for f in sorted(os.listdir(parkinson_dir))
                     if f.lower().endswith((".wav", ".mp3"))
                     and "_converted" not in f.lower()]
            print(f"  Processing {len(files)} parkinson audio files...")
            for f in files:
                path = os.path.join(parkinson_dir, f)
                emb = self.extract_embedding(path)
                if emb is not None:
                    X.append(emb)
                    y.append(1)

        if len(X) == 0:
            return np.array([]), np.array([])

        return np.array(X), np.array(y)

    def train(self, dataset_dir: str = "dataset/") -> "Wav2VecParkinsonModel":
        """
        Train the classification head on Wav2Vec2 embeddings.

        Args:
            dataset_dir: Root directory with healthy/ and parkinson/ subdirs.

        Returns:
            self (for chaining), or None if no data found.
        """
        print("\n-- Wav2Vec 2.0 Feature Extraction & Training --")
        X, y = self.load_dataset_from_directory(dataset_dir)

        if len(X) == 0:
            print("  No audio files found. Skipping Wav2Vec2 training.")
            return None

        print(f"  Training classifier on {len(X)} embeddings (shape: {X.shape})...")
        print(f"  Class distribution: healthy={np.sum(y == 0)}, parkinson={np.sum(y == 1)}")

        self.classifier.fit(X, y)

        # Report training accuracy (note: val accuracy from early_stopping is more reliable)
        train_preds = self.classifier.predict(X)
        train_acc = accuracy_score(y, train_preds)
        print(f"  Training accuracy: {train_acc:.4f}")

        if hasattr(self.classifier, "best_validation_score_"):
            print(f"  Best validation score: {self.classifier.best_validation_score_:.4f}")

        # Save classifier
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.classifier, self.classifier_path)
        print(f"  Classifier saved to: {self.classifier_path}")

        self._classifier_loaded = True
        return self

    def predict(self, audio_path: str) -> tuple:
        """
        Predict Parkinson's from an audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            (prediction, probability): prediction is 0 (healthy) or 1 (parkinson),
                                       probability is the confidence for class 1.
        """
        # Load classifier if not already loaded
        if not self._classifier_loaded:
            if not os.path.exists(self.classifier_path):
                raise FileNotFoundError(
                    f"Wav2Vec classifier not found at {self.classifier_path}. "
                    f"Run training first."
                )
            self.classifier = joblib.load(self.classifier_path)
            self._classifier_loaded = True

        # Extract embedding
        emb = self.extract_embedding(audio_path)
        if emb is None:
            raise ValueError(f"Could not extract embedding from {audio_path}")

        # Predict
        prediction = self.classifier.predict([emb])[0]

        if hasattr(self.classifier, "predict_proba"):
            probability = self.classifier.predict_proba([emb])[0][1]
        else:
            probability = 1.0 if prediction == 1 else 0.0

        return int(prediction), float(probability)


# ==============================================================================
# SMOKE TEST
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" WAV2VEC 2.0 PIPELINE SMOKE TEST")
    print("=" * 60)

    device = get_device()
    print(f"\n  Device: {device}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")

    # Test 1: Extract embeddings from random tensor
    print("\n--- Test 1: Random tensor input ---")
    random_audio = torch.randn(1, 48000)  # Simulates 3s of 16kHz audio
    print(f"  Input shape: {random_audio.shape}")

    try:
        embedding = extract_embeddings(random_audio)
        print(f"  ✅ Output shape: {embedding.shape}")
        print(f"     Expected:     torch.Size([1, 768])")
        print(f"     Match:        {'✅ YES' if embedding.shape == torch.Size([1, 768]) else '❌ NO'}")
        print(f"     Device:       {embedding.device}")
        print(f"     Dtype:        {embedding.dtype}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Extract from numpy array
    print("\n--- Test 2: Numpy array input ---")
    random_np = np.random.randn(48000).astype(np.float32)
    print(f"  Input shape: {random_np.shape}")

    try:
        embedding_np = extract_embeddings(random_np)
        print(f"  ✅ Output shape: {embedding_np.shape}")
        print(f"     Match:        {'✅ YES' if embedding_np.shape == torch.Size([1, 768]) else '❌ NO'}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # Test 3: Extract from real file (if available)
    print("\n--- Test 3: Real audio file ---")
    test_file = "test_healthy_voice.wav"
    if os.path.exists(test_file):
        try:
            embedding_file = extract_embeddings(test_file)
            print(f"  ✅ Output shape: {embedding_file.shape}")
            print(f"     Match:        {'✅ YES' if embedding_file.shape == torch.Size([1, 768]) else '❌ NO'}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    else:
        print(f"  Skipped — {test_file} not found.")

    # Test 4: Batch extraction
    print("\n--- Test 4: Batch extraction ---")
    batch_inputs = [torch.randn(1, 48000) for _ in range(3)]
    try:
        batch_emb = extract_embeddings_batch(batch_inputs)
        print(f"  ✅ Batch output shape: {batch_emb.shape}")
        print(f"     Expected:           torch.Size([3, 768])")
        print(f"     Match:              {'✅ YES' if batch_emb.shape == torch.Size([3, 768]) else '❌ NO'}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    print("\n" + "=" * 60)
    print(" SMOKE TEST COMPLETE")
    print("=" * 60)
