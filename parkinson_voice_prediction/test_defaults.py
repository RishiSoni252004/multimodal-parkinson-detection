import numpy as np
import joblib
from feature_extraction.extract_features import _set_defaults

features_dict = _set_defaults()
selected_features = joblib.load("models/feature_names.pkl")
feature_array = [features_dict.get(f, 0) for f in selected_features]

scaler = joblib.load("models/scaler.pkl")
scaled_features = scaler.transform(np.array(feature_array).reshape(1, -1))

model = joblib.load("models/best_model.pkl")
pred = model.predict(scaled_features)
prob = model.predict_proba(scaled_features)

print("Pred:", pred, "Prob:", prob)

