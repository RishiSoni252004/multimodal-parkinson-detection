import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from feature_extraction.extract_features import extract_features_from_audio

def load_pipeline():
    """Load model, scaler, and feature names, and optionally a background dataset for SHAP."""
    model_path = 'models/best_model.pkl'
    scaler_path = 'models/scaler.pkl'
    features_path = 'models/feature_names.pkl'
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        raise FileNotFoundError("Model files not found. Please train the model first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    
    # Try loading background data for SHAP if available
    background_data_path = 'models/background_data.pkl'
    background_data = None
    if os.path.exists(background_data_path):
        background_data = joblib.load(background_data_path)
        
    return model, scaler, feature_names, background_data

def generate_shap_plot(model, scaler, feature_names, background_data, instance_scaled, output_path):
    """Generates and saves a SHAP waterfall or bar plot for a single instance."""
    try:
        # Create explainer
        # If it's a tree model (Random Forest / XGBoost)
        model_name = type(model).__name__
        if model_name in ['RandomForestClassifier', 'XGBClassifier']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(instance_scaled)
            
            # Handle multiple outputs (for RF, shap_values might be a list where index 1 is positive class)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
                expected_val = explainer.expected_value[1]
            else:
                sv = shap_values[0]
                # XGBoost expected value handling
                expected_val = explainer.expected_value
                if isinstance(expected_val, (list, np.ndarray)):
                    expected_val = expected_val[0]
                
            explanation = shap.Explanation(values=sv, base_values=expected_val, data=instance_scaled[0], feature_names=feature_names)
            
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(explanation, show=False)
        else:
            # SVM or other model, fallback to KernelExplainer if background data is available
            if background_data is not None:
                # Need probability predict function for KernelExplainer
                predict_fn = lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x)
                explainer = shap.KernelExplainer(predict_fn, background_data)
                shap_values = explainer.shap_values(instance_scaled, nsamples=100)
                
                # shap_values could be a 1D array for single instance
                sv = shap_values[0] if len(np.shape(shap_values)) > 1 else shap_values
                expected_val = explainer.expected_value
                
                plt.figure(figsize=(10, 6))
                # For kernel explainer we can use summary plot as bar
                shap.summary_plot(np.array([sv]), features=instance_scaled, feature_names=feature_names, plot_type="bar", show=False)
            else:
                print("No background data for SVM SHAP. Skipping SHAP plot.")
                return None
                
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    except Exception as e:
        print(f"Error generating SHAP plot: {e}")
        return None

def predict_audio(file_path, output_shap_dir="frontend/static/plots"):
    """
    Predicts Parkinson's from an audio file and generates a SHAP plot.
    """
    try:
        model, scaler, feature_names, background_data = load_pipeline()
    except FileNotFoundError as e:
        return {"error": str(e)}

    # Extract features
    features = extract_features_from_audio(file_path)
    if not features:
        return {"error": "Failed to extract features from audio."}
        
    # Order features correctly
    try:
        feature_vector = [features[name] for name in feature_names]
    except KeyError as e:
        return {"error": f"Missing feature: {e}. Ensure model was trained with the same features."}
        
    X_instance = np.array(feature_vector).reshape(1, -1)
    X_instance_scaled = scaler.transform(X_instance)
    
    # Predict
    prediction = model.predict(X_instance_scaled)[0]
    
    # Get probability
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_instance_scaled)[0]
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
    else:
        confidence = 1.0 # Fallback if no predict_proba
        
    # Generate SHAP plot
    os.makedirs(output_shap_dir, exist_ok=True)
    base_name = os.path.basename(file_path).split('.')[0]
    shap_path = os.path.join(output_shap_dir, f"shap_{base_name}.png")
    
    generate_shap_plot(model, scaler, feature_names, background_data, X_instance_scaled, shap_path)
    
    result_text = "Parkinson's Detected" if prediction == 1 else "Healthy (No Parkinson's Detected)"
    
    return {
        "prediction": result_text,
        "confidence": float(confidence),
        "prediction_class": int(prediction),
        "shap_plot_path": shap_path,
        "features": features
    }
