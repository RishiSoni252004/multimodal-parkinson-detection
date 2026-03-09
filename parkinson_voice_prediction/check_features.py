import numpy as np
import joblib
from feature_extraction.extract_features import extract_features_from_audio

# Load model pipeline
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

file_path = 'dataset/temp_uploads/recording.webm'
features = extract_features_from_audio(file_path)

if features:
    feature_vector = [features[name] for name in feature_names]
    X_instance = np.array(feature_vector).reshape(1, -1)
    
    # Check original distribution
    print("--- Extracted Features ---")
    print(X_instance)
    
    # Check scaled distribution
    X_instance_scaled = scaler.transform(X_instance)
    print("\n--- Scaled Features ---")
    print(X_instance_scaled)
    
    # Check model probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_instance_scaled)
        print("\n--- Prediction Probabilities ---")
        print(probs)
    
    prediction = model.predict(X_instance_scaled)
    print("\n--- Prediction ---")
    print(prediction)
else:
    print("Extraction failed")
