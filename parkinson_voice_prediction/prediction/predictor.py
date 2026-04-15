import os
import joblib
import numpy as np
import subprocess
from feature_extraction.extract_features import extract_features_from_audio
from models.wav2vec_model import Wav2VecParkinsonModel
from models.spiral_model import SpiralModel

def convert_to_wav(input_path):
    """
    Convert any audio file to WAV at 16000Hz mono using ffmpeg.
    This resolves issues with browser recordings missing headers or 
    having incompatible sample rates for Praat/Librosa.
    """
    output_path = os.path.splitext(input_path)[0] + '_converted.wav'
    print(f"Converting {input_path} to {output_path}...")
    try:
        res = subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-ar', '16000',   # 16kHz sample rate (better for voice features generally)
            '-ac', '1',       # mono
            output_path
        ], capture_output=True, text=True, timeout=30)
        
        if res.returncode != 0:
            raise RuntimeError(f"FFMPEG returned non-zero code. Error: {res.stderr}")
            
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            raise RuntimeError("FFMPEG completed but output file is empty/missing.")
            
    except Exception as e:
        print(f"ffmpeg conversion error: {e}")
        # Return the original file, it might just be valid enough for librosa
    
    return input_path

class Predictor:
    def __init__(self):
        self.scaler_path = "models/scaler.pkl"
        self.selected_features_path = "models/selected_features.pkl"
        self.classical_model_path = "models/best_model.pkl"
        self.wav2vec_classifier_path = "models/wav2vec_classifier.pkl"
        self.spiral_model_path = "models/spiral_model.pth"
        self.spiral_model = None
        
    def predict_from_features(self, feature_array, use_wav2vec=False):
        """Option 1: Accept voice feature values directly."""
        if use_wav2vec:
            raise ValueError("Wav2Vec requires raw audio file, not feature arrays.")
            
        if not os.path.exists(self.classical_model_path):
            raise FileNotFoundError(f"Classical model not found at {self.classical_model_path}.")
            
        scaler = joblib.load(self.scaler_path)
        model = joblib.load(self.classical_model_path)
        
        # Scale
        scaled_features = scaler.transform(np.array(feature_array).reshape(1, -1))
        
        prediction = model.predict(scaled_features)[0]
        prob = model.predict_proba(scaled_features)[0][1] if hasattr(model, "predict_proba") else 1.0
        
        return "Parkinson Detected" if prediction == 1 else "Healthy", prob

    def predict_from_audio(self, audio_path, use_wav2vec=False):
        """Option 2: Accept a voice audio file."""
        
        # Format the audio to standard wav
        clean_audio_path = convert_to_wav(audio_path)
        
        if use_wav2vec:
            wav2vec_model = Wav2VecParkinsonModel()
            if not os.path.exists(self.wav2vec_classifier_path):
                raise FileNotFoundError("Wav2Vec classification head not found.")
                
            prediction, prob = wav2vec_model.predict(clean_audio_path)
            return "Parkinson Detected" if prediction == 1 else "Healthy", float(prob)
            
        else:
            # Extract features manually
            features_dict = extract_features_from_audio(clean_audio_path)
            if not features_dict:
                raise ValueError("Failed to extract classical features from audio.")
                
            selected_features = joblib.load(self.selected_features_path)
            # Default to 0 if feature missing
            feature_array = [features_dict.get(f, 0) for f in selected_features]
            
            return self.predict_from_features(feature_array)

    def predict_from_spiral_image(self, image_path):
        """Option 3: Accept a spiral drawing image file."""
        if self.spiral_model is None:
            self.spiral_model = SpiralModel(self.spiral_model_path)
            
        return self.spiral_model.predict(image_path)
