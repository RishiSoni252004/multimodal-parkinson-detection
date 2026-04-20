"""
preprocess.py — Reusable Audio Preprocessing for Parkinson's Detection

Standardizes any audio input to a consistent format:
  - Sample rate: 16kHz
  - Channels: Mono
  - Duration: Exactly 3 seconds (pad with zeros or trim)
  - Amplitude: Normalized to [-1, 1]
  - Output: torch.Tensor of shape [1, 48000]

Uses librosa as primary loader with torchaudio as fallback.
"""

import os
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")

# Try importing audio libraries
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


# ==============================================================================
# CONFIGURATION
# ==============================================================================

TARGET_SR = 16000          # 16kHz — standard for speech models & Wav2Vec2
TARGET_DURATION = 3.0      # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 48000 samples
TARGET_CHANNELS = 1        # mono
TRIM_DB = 20               # silence threshold for trimming (dB)


# ==============================================================================
# CORE PREPROCESSING FUNCTION
# ==============================================================================

def preprocess_audio(path: str) -> torch.Tensor:
    """
    Load and preprocess an audio file into a standardized torch.Tensor.

    Pipeline:
      1. Load audio (librosa primary, torchaudio fallback)
      2. Convert to mono if stereo
      3. Resample to 16kHz
      4. Trim leading/trailing silence
      5. Normalize amplitude to [-1, 1]
      6. Pad with zeros or trim to exactly 3 seconds (48000 samples)
      7. Return as torch.Tensor of shape [1, 48000]

    Args:
        path: Absolute or relative path to an audio file (.wav, .mp3, .webm, etc.)

    Returns:
        torch.Tensor: Shape [1, 48000] — mono audio at 16kHz, 3 seconds.

    Raises:
        FileNotFoundError: If the audio file doesn't exist.
        RuntimeError: If neither librosa nor torchaudio can load the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio = None

    # === Strategy 1: librosa (preferred) ===
    if HAS_LIBROSA:
        try:
            # librosa.load automatically:
            #   - Converts to mono (mono=True by default)
            #   - Resamples to target SR
            #   - Returns float32 numpy array
            audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
        except Exception as e:
            print(f"  librosa failed: {e}")
            audio = None

    # === Strategy 2: torchaudio (fallback) ===
    if audio is None and HAS_TORCHAUDIO:
        try:
            waveform, sr = torchaudio.load(path)

            # Convert to mono by averaging channels
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=TARGET_SR
                )
                waveform = resampler(waveform)

            # Convert to numpy for consistent processing
            audio = waveform.squeeze(0).numpy()
        except Exception as e:
            print(f"  torchaudio failed: {e}")
            audio = None

    # === Neither library worked ===
    if audio is None:
        libs = []
        if not HAS_LIBROSA:
            libs.append("librosa")
        if not HAS_TORCHAUDIO:
            libs.append("torchaudio")
        if libs:
            raise RuntimeError(
                f"Cannot load audio. Missing libraries: {', '.join(libs)}. "
                f"Install with: pip install librosa torchaudio"
            )
        else:
            raise RuntimeError(
                f"Both librosa and torchaudio failed to load: {path}. "
                f"File may be corrupted or in an unsupported format."
            )

    # === Post-processing pipeline ===

    # Step 1: Ensure float32
    audio = audio.astype(np.float32)

    # Step 2: Trim silence from beginning and end
    if HAS_LIBROSA:
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=TRIM_DB)
            if len(trimmed) > 0:
                audio = trimmed
        except Exception:
            pass  # If trimming fails, use original

    # Step 3: Normalize amplitude to [-1, 1]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    # If max_val is 0, audio is silence — leave as zeros

    # Step 4: Pad or trim to exactly TARGET_SAMPLES (48000)
    if len(audio) > TARGET_SAMPLES:
        # Trim from the middle (center crop) to keep the most relevant part
        start = (len(audio) - TARGET_SAMPLES) // 2
        audio = audio[start : start + TARGET_SAMPLES]
    elif len(audio) < TARGET_SAMPLES:
        # Pad with zeros at the end
        padding = TARGET_SAMPLES - len(audio)
        audio = np.pad(audio, (0, padding), mode="constant", constant_values=0.0)

    # Step 5: Convert to torch.Tensor with shape [1, 48000]
    tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    return tensor


# ==============================================================================
# BATCH PREPROCESSING
# ==============================================================================

def preprocess_audio_batch(paths: list) -> torch.Tensor:
    """
    Preprocess a batch of audio files.

    Args:
        paths: List of file paths to audio files.

    Returns:
        torch.Tensor: Shape [batch_size, 1, 48000]
    """
    tensors = []
    for path in paths:
        try:
            t = preprocess_audio(path)
            tensors.append(t)
        except Exception as e:
            print(f"  Skipping {path}: {e}")

    if not tensors:
        raise ValueError("No audio files could be preprocessed.")

    return torch.stack(tensors)


# ==============================================================================
# UTILITY: Get audio info without full preprocessing
# ==============================================================================

def get_audio_info(path: str) -> dict:
    """
    Get basic info about an audio file without full preprocessing.

    Args:
        path: Path to audio file.

    Returns:
        dict with keys: duration, sample_rate, channels, samples
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    if HAS_LIBROSA:
        y, sr = librosa.load(path, sr=None, mono=False)
        if y.ndim == 1:
            channels = 1
            samples = len(y)
        else:
            channels = y.shape[0]
            samples = y.shape[1]
        return {
            "duration": samples / sr,
            "sample_rate": sr,
            "channels": channels,
            "samples": samples,
        }
    elif HAS_TORCHAUDIO:
        info = torchaudio.info(path)
        return {
            "duration": info.num_frames / info.sample_rate,
            "sample_rate": info.sample_rate,
            "channels": info.num_channels,
            "samples": info.num_frames,
        }
    else:
        raise RuntimeError("Neither librosa nor torchaudio is available.")


# ==============================================================================
# SELF-TEST
# ==============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print(" AUDIO PREPROCESSING SELF-TEST")
    print("=" * 60)

    print(f"\n  Libraries:")
    print(f"    librosa:    {'✅ available' if HAS_LIBROSA else '❌ missing'}")
    print(f"    torchaudio: {'✅ available' if HAS_TORCHAUDIO else '❌ missing'}")

    print(f"\n  Target format:")
    print(f"    Sample rate: {TARGET_SR} Hz")
    print(f"    Duration:    {TARGET_DURATION} s")
    print(f"    Samples:     {TARGET_SAMPLES}")
    print(f"    Channels:    {TARGET_CHANNELS} (mono)")

    # Find a test file
    test_files = [
        "test_healthy_voice.wav",
        "dataset/healthy/sample_0.wav",
    ]

    test_file = None
    for tf in test_files:
        if os.path.exists(tf):
            test_file = tf
            break

    if test_file is None:
        print("\n  ❌ No test audio file found. Skipping file test.")
        print("     Place a .wav file as 'test_healthy_voice.wav' to test.")
    else:
        print(f"\n  Test file: {test_file}")

        # Get original info
        try:
            info = get_audio_info(test_file)
            print(f"  Original: sr={info['sample_rate']}Hz, "
                  f"duration={info['duration']:.2f}s, "
                  f"channels={info['channels']}, "
                  f"samples={info['samples']}")
        except Exception as e:
            print(f"  Could not read info: {e}")

        # Preprocess
        print(f"\n  Running preprocess_audio('{test_file}')...")
        try:
            result = preprocess_audio(test_file)

            print(f"\n  ✅ Result shape: {result.shape}")
            print(f"     Expected:     torch.Size([1, {TARGET_SAMPLES}])")
            print(f"     Match:        {'✅ YES' if result.shape == torch.Size([1, TARGET_SAMPLES]) else '❌ NO'}")
            print(f"     Dtype:        {result.dtype}")
            print(f"     Min value:    {result.min().item():.6f}")
            print(f"     Max value:    {result.max().item():.6f}")
            print(f"     Mean value:   {result.mean().item():.6f}")

            # Verify amplitude normalization
            assert result.min() >= -1.0, "Min value below -1.0!"
            assert result.max() <= 1.0, "Max value above 1.0!"
            assert result.shape == torch.Size([1, TARGET_SAMPLES]), "Shape mismatch!"
            print(f"\n  ✅ All assertions passed!")

        except Exception as e:
            print(f"\n  ❌ Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()

    # Test with a random tensor (simulating raw audio)
    print(f"\n  --- Synthetic Audio Test ---")
    # Create a fake 5-second audio file to test trimming
    fake_audio = np.random.randn(TARGET_SR * 5).astype(np.float32) * 0.5
    fake_path = "dataset/temp_uploads/_test_synthetic.wav"
    os.makedirs("dataset/temp_uploads", exist_ok=True)

    if HAS_LIBROSA:
        import soundfile as sf
        sf.write(fake_path, fake_audio, TARGET_SR)
        print(f"  Created synthetic audio: {fake_path} (5 seconds)")

        result = preprocess_audio(fake_path)
        print(f"  Result shape: {result.shape}")
        print(f"  Match: {'✅ YES' if result.shape == torch.Size([1, TARGET_SAMPLES]) else '❌ NO'}")

        # Cleanup
        os.remove(fake_path)
        print(f"  Cleaned up test file.")

    print("\n" + "=" * 60)
    print(" SELF-TEST COMPLETE")
    print("=" * 60)
