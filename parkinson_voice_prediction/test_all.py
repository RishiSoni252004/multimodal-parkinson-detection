from prediction.predictor import Predictor
import os
import glob

print("--- Testing Initialization ---")
p = Predictor()

audio_file = glob.glob("dataset/healthy/*.wav")[0]

print("\n--- Testing Wav2Vec ---")
try:
    res, prob = p.predict_from_audio(audio_file, use_wav2vec=True)
    print(f"Wav2Vec Success: {res} ({prob:.4f})")
except Exception as e:
    print(f"Wav2Vec FAILED: {e}")

print("\n--- Testing VoiceFNN ---")
try:
    res, prob = p.predict_from_audio(audio_file, use_wav2vec=False)
    print(f"VoiceFNN Success: {res} ({prob:.4f})")
except Exception as e:
    print(f"VoiceFNN FAILED: {e}")

print("\n--- Testing Classical ML ---")
try:
    res, prob = p.predict_from_features([0]*16)
    print(f"Classical ML Success: {res} ({prob:.4f})")
except Exception as e:
    print(f"Classical ML FAILED: {e}")

print("\n--- Testing Spiral ---")
if not os.path.exists("test_spiral.png"):
    import numpy as np
    from PIL import Image
    imarray = np.random.rand(100,100,3) * 255
    im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    im.save('test_spiral.png')

try:
    res, prob = p.predict_from_spiral_image("test_spiral.png")
    print(f"Spiral Success: {res}")
except Exception as e:
    print(f"Spiral FAILED: {e}")

