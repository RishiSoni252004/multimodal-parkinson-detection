from feature_extraction.extract_features import extract_features_from_audio
import joblib
features_dict = extract_features_from_audio("dataset/healthy/AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav")
print(features_dict)
print(joblib.load("models/feature_names.pkl"))
