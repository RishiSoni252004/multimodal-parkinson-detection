import os
import glob
import librosa
import soundfile as sf
from PIL import Image

def clean_audio():
    print("--- Cleaning Audio Files ---")
    audio_files = glob.glob("dataset/healthy/*.wav") + glob.glob("dataset/parkinson/*.wav")
    for file in audio_files:
        try:
            # Load audio preserving original sample rate
            y, sr = librosa.load(file, sr=None)
            
            # Trim leading and trailing silence (below 20 decibels)
            y_trimmed, index = librosa.effects.trim(y, top_db=20)
            
            # Overwrite with the cleaned, trimmed version
            sf.write(file, y_trimmed, sr)
            print(f"Cleaned & Trimmed: {file}")
        except Exception as e:
            print(f"Failed to clean {file}: {e}")

def clean_images():
    print("--- Cleaning Image Files ---")
    image_files = glob.glob("dataset/spiral/healthy/*.*") + glob.glob("dataset/spiral/parkinson/*.*")
    for file in image_files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(file)
                
                # Convert to RGB to ensure 3-channels (drops Alpha channels or fixes Grayscale)
                img = img.convert('RGB')
                
                # Resize to standard size for our ResNet Model (224x224)
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                
                # Overwrite original
                img.save(file)
                print(f"Cleaned & Standardized: {file}")
            except Exception as e:
                print(f"Failed to clean {file}: {e}")

if __name__ == "__main__":
    clean_audio()
    clean_images()
    print("\n✅ Dataset cleaning completed successfully.")
