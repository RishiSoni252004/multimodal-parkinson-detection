import sys
import wave
import contextlib
import numpy as np

# Create a dummy healthy-like audio recording automatically to test
# 16000Hz, 3 seconds of a sine wave (very clean, no jitter/shimmer -> should be healthy)
sr = 44100
duration = 3.0
t = np.linspace(0, duration, int(sr * duration), False)
# Generate a 120Hz sine wave (healthy human pitch)
audio = np.sin(120 * 2 * np.pi * t) * 0.5
# convert to 16bit PCM
audio_pcm = np.int16(audio * 32767)

with wave.open("dataset/healthy/AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav", "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sr)
    f.writeframes(audio_pcm.tobytes())

from prediction.predictor import Predictor
p = Predictor()

try:
    print("Testing Classical Pipeline with Praat...")
    result, prob = p.predict_from_audio("dataset/healthy/AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav")
    print(f"Result: {result}, Prob: {prob}")
except Exception as e:
    print(f"Classical ML Error: {e}")

try:
    print("\nTesting DL Pipeline with resnet/FNN...")
    # By forcing an exception in the ML one, or just calling predict_audio directly
    from prediction.predictor import predict_audio
    result, prob = predict_audio("dataset/healthy/AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav")
    print(f"Result: {result}, Prob: {prob}")
except Exception as e:
    print(f"DL Error: {e}")

