"""
predictor.py — Central Prediction Orchestrator for Parkinson's Detection

Routes predictions to the appropriate model:
  1. Clinical features → Classical ML (sklearn)
  2. Audio (wav2vec) → Wav2Vec2 embeddings → PyTorch classifier
  3. Audio (DL) → MFCC features → VoiceFNN
  4. Spiral image → ResNet-18 CNN

All device handling uses MPS (Apple Silicon) > CPU. Never CUDA.
"""

import os
import sys
import joblib
import numpy as np
import subprocess
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extraction.extract_features import extract_features_from_audio
from feature_extraction.extract_dl_features import VoiceDataProcessor
from models.spiral_model import SpiralModel
from models.voice_dl_model import VoiceFNN


def get_device() -> torch.device:
    """Get best available device: MPS > CPU. Never CUDA."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio file to WAV at 16000Hz mono using ffmpeg.
    Falls back to librosa-based conversion if ffmpeg is not available.

    Args:
        input_path: Path to input audio file.

    Returns:
        Path to converted WAV file, or original path if conversion not needed/failed.
    """
    # If already a standard wav, try using it directly
    if input_path.lower().endswith('.wav'):
        # Still try to standardize it via librosa (no ffmpeg needed)
        try:
            import librosa
            import soundfile as sf

            y, sr = librosa.load(input_path, sr=16000, mono=True)
            output_path = os.path.splitext(input_path)[0] + '_converted.wav'
            sf.write(output_path, y, 16000)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except Exception:
            pass
        return input_path

    # For non-wav formats, try ffmpeg first, then librosa
    output_path = os.path.splitext(input_path)[0] + '_converted.wav'

    # Try ffmpeg
    try:
        res = subprocess.run(
            ['ffmpeg', '-y', '-i', input_path,
             '-ar', '16000', '-ac', '1', output_path],
            capture_output=True, text=True, timeout=30
        )
        if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except (FileNotFoundError, Exception):
        pass

    # Try librosa fallback
    try:
        import librosa
        import soundfile as sf

        y, sr = librosa.load(input_path, sr=16000, mono=True)
        sf.write(output_path, y, 16000)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except Exception as e:
        print(f"  Audio conversion failed: {e}")

    return input_path


def predict_audio_dl(file_path: str) -> tuple:
    """
    Predict using the Voice Deep Learning PyTorch model (MFCC features + VoiceFNN).

    Args:
        file_path: Path to audio file.

    Returns:
        (prediction_label, probability)
    """
    device = get_device()

    processor = VoiceDataProcessor(target_sr=16000, duration=3.0)
    y = processor.preprocess_audio(file_path)
    if y is None or len(y) == 0:
        raise ValueError("Audio is empty or corrupted.")

    features = processor.extract_features(y)

    scaler_path = "models/voice_dl_scaler.pkl"
    model_path = "models/voice_dl_model.pth"

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError("DL Voice model or scaler not found. Run training pipeline.")

    scaler = joblib.load(scaler_path)
    scaled_features = scaler.transform(np.array(features).reshape(1, -1))

    input_size = scaled_features.shape[1]
    model = VoiceFNN(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = torch.tensor(scaled_features, dtype=torch.float32).to(device)
        outputs = model(inputs)
        # Temperature scaling (T=2.0) to soften extreme confidences
        probs = torch.softmax(outputs / 2.0, dim=1)
        prob_parkinson = probs[0][1].item()

    prediction = "Parkinson Detected" if prob_parkinson >= 0.5 else "Healthy"
    return prediction, float(prob_parkinson)


class Predictor:
    """Central prediction class used by the Streamlit app."""

    def __init__(self):
        self.scaler_path = "models/scaler.pkl"
        self.selected_features_path = "models/selected_features.pkl"
        self.classical_model_path = "models/best_model.pkl"
        self.wav2vec_classifier_path = "checkpoints/voice_best.pt"
        self.spiral_model_path = "models/spiral_model.pth"
        self.spiral_model = None

    def predict_from_features(self, feature_array: list, use_wav2vec: bool = False) -> tuple:
        """
        Predict from pre-extracted clinical voice feature values.

        Args:
            feature_array: List of float feature values.
            use_wav2vec: Must be False for this method.

        Returns:
            (prediction_label, probability)
        """
        if use_wav2vec:
            raise ValueError("Wav2Vec requires raw audio file, not feature arrays.")

        if not os.path.exists(self.classical_model_path):
            raise FileNotFoundError(f"Classical model not found at {self.classical_model_path}.")

        scaler = joblib.load(self.scaler_path)
        model = joblib.load(self.classical_model_path)

        scaled_features = scaler.transform(np.array(feature_array).reshape(1, -1))
        prediction = model.predict(scaled_features)[0]
        prob = model.predict_proba(scaled_features)[0][1] if hasattr(model, "predict_proba") else 1.0

        return "Parkinson Detected" if prediction == 1 else "Healthy", prob

    def predict_from_audio(self, audio_path: str, use_wav2vec: bool = False) -> tuple:
        """
        Predict from an audio file.

        Args:
            audio_path: Path to audio file (.wav, .mp3, .webm).
            use_wav2vec: If True, use Wav2Vec2 + PyTorch classifier. Otherwise use DL voice model.

        Returns:
            (prediction_label, probability)
        """
        # Standardize audio format
        clean_audio_path = convert_to_wav(audio_path)

        if use_wav2vec:
            # === Wav2Vec2 + PyTorch Classifier ===
            try:
                from models.voice_classifier import predict_with_classifier
                label, confidence = predict_with_classifier(
                    clean_audio_path,
                    checkpoint_path=self.wav2vec_classifier_path
                )
                prob = confidence if label == "Parkinson Detected" else (1.0 - confidence)
                return label, float(prob)
            except FileNotFoundError:
                # Fall back to sklearn-based wav2vec classifier
                try:
                    from models.wav2vec_model import Wav2VecParkinsonModel
                    wav2vec_model = Wav2VecParkinsonModel()
                    prediction, prob = wav2vec_model.predict(clean_audio_path)
                    return "Parkinson Detected" if prediction == 1 else "Healthy", float(prob)
                except Exception as e:
                    raise RuntimeError(f"Wav2Vec prediction failed: {e}")
        else:
            # === DL Voice Model (MFCC features) ===
            try:
                prediction, prob = predict_audio_dl(clean_audio_path)
                return prediction, prob
            except Exception as dl_error:
                print(f"  DL Model failed, falling back to classical ML... ({dl_error})")

                # === Fallback: Classical ML ===
                try:
                    features_dict = extract_features_from_audio(clean_audio_path)
                    if not features_dict:
                        raise ValueError("Failed to extract features from audio.")

                    selected_features = joblib.load(self.selected_features_path)
                    feature_array = [features_dict.get(f, 0) for f in selected_features]
                    return self.predict_from_features(feature_array)
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"All prediction methods failed.\n"
                        f"  DL error: {dl_error}\n"
                        f"  Fallback error: {fallback_error}"
                    )

    def predict_from_spiral_image(self, image_path: str) -> tuple:
        """
        Predict from a spiral drawing image.

        Args:
            image_path: Path to spiral drawing (.png, .jpg, .jpeg).

        Returns:
            (prediction_label, probability)
        """
        if self.spiral_model is None:
            self.spiral_model = SpiralModel(self.spiral_model_path)

        return self.spiral_model.predict(image_path)
