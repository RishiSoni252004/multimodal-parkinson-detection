import numpy as np
from feature_extraction.extract_features import extract_features_from_audio
import joblib

features = extract_features_from_audio("test_healthy_voice.wav")
print("Features extracted:", features)

selected_features = joblib.load("models/selected_features.pkl")
print("Selected features:", selected_features)
print("Keys in features:", list(features.keys()))

feature_array = [features.get(f, 0) for f in selected_features]
print("Feature array:", feature_array)

scaler = joblib.load("models/scaler.pkl")
scaled_features = scaler.transform(np.array(feature_array).reshape(1, -1))
print("Scaled features:", scaled_features)

model = joblib.load("models/best_model.pkl")
pred = model.predict(scaled_features)
prob = model.predict_proba(scaled_features)

print("Pred:", pred, "Prob:", prob)

