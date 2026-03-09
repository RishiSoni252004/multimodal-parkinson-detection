import os
import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

class Wav2VecParkinsonModel:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.model.eval()
        self.classifier = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        
    def extract_embedding(self, audio_path):
        try:
            # Wav2Vec2 expects 16kHz audio
            speech, sr = librosa.load(audio_path, sr=16000)
            
            # Truncate at 10 seconds to avoid memory overflow on very long samples
            max_len = 16000 * 10 
            if len(speech) > max_len:
                speech = speech[:max_len]
                
            input_values = self.processor(speech, return_tensors="pt", sampling_rate=16000).input_values
            
            with torch.no_grad():
                outputs = self.model(input_values)
                # Average pooling over the sequence length axis
                hidden_states = outputs.last_hidden_state
                embedding = torch.mean(hidden_states, dim=1).squeeze().numpy()
                return embedding
        except Exception as e:
            print(f"Error extracting embedding from {audio_path}: {e}")
            return None

    def load_dataset_from_directory(self, root_dir):
        X, y = [], []
        # Expected mapping: healthy=0 or 1. Let's map target to 1=Parkinsons
        # Note: classical target is 1 for Parkinsons, 0 for healthy.
        if os.path.exists(os.path.join(root_dir, 'healthy')):
            for f in os.listdir(os.path.join(root_dir, 'healthy')):
                path = os.path.join(root_dir, 'healthy', f)
                if path.lower().endswith('.wav') or path.lower().endswith('.mp3'):
                    print(f"Extracting wav2vec features for {path}")
                    emb = self.extract_embedding(path)
                    if emb is not None:
                        X.append(emb)
                        y.append(0)
                        
        if os.path.exists(os.path.join(root_dir, 'parkinson')):
            for f in os.listdir(os.path.join(root_dir, 'parkinson')):
                path = os.path.join(root_dir, 'parkinson', f)
                if path.lower().endswith('.wav') or path.lower().endswith('.mp3'):
                    print(f"Extracting wav2vec features for {path}")
                    emb = self.extract_embedding(path)
                    if emb is not None:
                        X.append(emb)
                        y.append(1)
        
        return np.array(X), np.array(y)

    def train(self, dataset_dir="dataset/"):
        print("-- Wav2Vec 2.0 Feature Extraction & Training --")
        X, y = self.load_dataset_from_directory(dataset_dir)
        if len(X) == 0:
            print("No audio files found for Wav2Vec2 training in 'dataset/healthy' and 'dataset/parkinson'. Skipping.")
            return None
            
        print(f"Training classification head on {len(X)} audio embeddings...")
        self.classifier.fit(X, y)
        preds = self.classifier.predict(X)
        print(f"Wav2Vec2 Training Accuracy: {accuracy_score(y, preds):.4f}")
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.classifier, "models/wav2vec_classifier.pkl")
        print("Saved Wav2Vec classification head to models/wav2vec_classifier.pkl")
        return self

    def predict(self, audio_path):
        if not os.path.exists("models/wav2vec_classifier.pkl"):
            raise FileNotFoundError("Wav2Vec model head is not trained.")
            
        clf = joblib.load("models/wav2vec_classifier.pkl")
        emb = self.extract_embedding(audio_path)
        if emb is None:
            raise ValueError("Could not extract embedding")
        return clf.predict([emb])[0], clf.predict_proba([emb])[0][1] if hasattr(clf, "predict_proba") else 1.0
