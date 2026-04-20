from prediction.predictor import Predictor
import os
import glob
import numpy as np

print("--- Testing Initialization ---")
p = Predictor()

audio_file = glob.glob("dataset/healthy/*.wav")[0]
parkinson_file = glob.glob("dataset/parkinson/*.wav")[0]

print("\n--- Testing Wav2Vec ---")
res, prob = p.predict_from_audio(audio_file, use_wav2vec=True)
print(f"Wav2Vec Healthy Audio -> {res} ({prob:.4f})")
res, prob = p.predict_from_audio(parkinson_file, use_wav2vec=True)
print(f"Wav2Vec Parkinson Audio -> {res} ({prob:.4f})")

print("\n--- Testing DL VoiceFNN ---")
res, prob = p.predict_from_audio(audio_file, use_wav2vec=False)
print(f"DL Voice Healthy Audio -> {res} ({prob:.4f})")
res, prob = p.predict_from_audio(parkinson_file, use_wav2vec=False)
print(f"DL Voice Parkinson Audio -> {res} ({prob:.4f})")

print("\n--- Testing Classical ML ---")
# Give it dummy good data to ensure pipeline executes predicting healthy/parkinsons without crashing
# High Jitter/Shimmer vector -> Parkinsons
res, prob = p.predict_from_features([200, 250, 100, 0.03, 0.03, 0.03, 0.03, 0.03, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02, 10])
print(f"Classical ML (Bad Features) -> {res} ({prob:.4f})")
# Low Jitter/Shimmer vector -> Healthy
res, prob = p.predict_from_features([200, 250, 100, 0.001, 0.001, 0.001, 0.001, 0.001, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.002, 35])
print(f"Classical ML (Good Features) -> {res} ({prob:.4f})")

print("\n--- Testing Spiral ---")
res, prob = p.predict_from_spiral_image("dataset/spiral/healthy/Healthy223.png")
print(f"Spiral Healthy Image -> {res} ({prob:.4f})")
res, prob = p.predict_from_spiral_image("dataset/spiral/parkinson/Parkinson40.png")
print(f"Spiral Parkinson Image -> {res} ({prob:.4f})")

