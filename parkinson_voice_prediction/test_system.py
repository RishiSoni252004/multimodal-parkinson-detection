import os
import joblib
import numpy as np
from prediction.predictor import Predictor
from feature_extraction.extract_features import extract_features_from_audio

def test_pipeline():
    print("--- Testing Pipeline Components ---")
    
    # Check if models exist
    print("\n1. Model Loading Check:")
    pred = Predictor()
    scaler = joblib.load(pred.scaler_path)
    model = joblib.load(pred.classical_model_path)
    features_list = joblib.load(pred.selected_features_path)
    
    print(f"Classical Model: {type(model).__name__}")
    print(f"Scaler: {type(scaler).__name__}")
    print(f"Expected Features Length: {len(features_list)}")
    print(f"Features: {features_list}")
    
    # Check default extraction (fallback)
    print("\n2. Default Fallback Extraction test (what happens if audio fails):")
    try:
        from feature_extraction.extract_features import _set_defaults
        defaults = _set_defaults()
        default_array = [defaults.get(f, 0) for f in features_list]
        
        scaled_def = scaler.transform(np.array(default_array).reshape(1, -1))
        prediction = model.predict(scaled_def)[0]
        prob = model.predict_proba(scaled_def)[0][1] if hasattr(model, "predict_proba") else 1.0
        
        print(f"Fallback scaled values: {scaled_def[0][:5]}...")
        print(f"Fallback prediction: {'Parkinson Detected' if prediction == 1 else 'Healthy'} (Prob: {prob:.4f})")
    except Exception as e:
        print(f"Error testing defaults: {e}")
        
    print("\n3. Testing with sample audio files:")
    healthy_sample = "dataset/healthy/sample_0.wav"
    park_sample = "dataset/parkinson/sample_0.wav"
    
    for label, path in [("Healthy File", healthy_sample), ("Parkinson File", park_sample)]:
        if os.path.exists(path):
            print(f"\n--- Testing {label} ({path}) ---")
            features = extract_features_from_audio(path)
            feature_array = [features.get(f, 0) for f in features_list]
            
            scaled_feat = scaler.transform(np.array(feature_array).reshape(1, -1))
            prediction = model.predict(scaled_feat)[0]
            prob = model.predict_proba(scaled_feat)[0][1] if hasattr(model, "predict_proba") else 1.0
            
            print(f"Raw Features (first 5): {feature_array[:5]}")
            print(f"Scaled Features (first 5): {scaled_feat[0][:5]}")
            print(f"Prediction: {'Parkinson Detected' if prediction == 1 else 'Healthy'} (Prob: {prob:.4f})")
        else:
            print(f"Missing sample: {path}")

if __name__ == "__main__":
    test_pipeline()
