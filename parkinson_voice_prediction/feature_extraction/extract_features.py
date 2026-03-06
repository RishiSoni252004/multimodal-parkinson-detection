import os
import librosa
import numpy as np
import pandas as pd

def extract_features_from_audio(file_path):
    """
    Extracts voice features from a .wav audio file using librosa.
    Features: MFCC, Pitch, Spectral Centroid, Zero Crossing Rate, Chroma Features.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        
        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        cent_mean = np.mean(spectral_centroids)
        
        # Zero Crossing Rate
        zero_crossing_rates = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zero_crossing_rates)
        
        # Chroma Features
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
        chroma_mean = np.mean(chromagram.T, axis=0)
        
        # Pitch (Fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        # Get mean pitch of significant magnitudes
        pitches_valid = pitches[magnitudes > np.median(magnitudes)]
        pitch_mean = np.mean(pitches_valid) if len(pitches_valid) > 0 else 0
        
        features = {}
        for i, val in enumerate(mfcc_mean):
            features[f'mfcc_{i}'] = val
        features['spectral_centroid'] = cent_mean
        features['zero_crossing_rate'] = zcr_mean
        features['pitch'] = pitch_mean
        for i, val in enumerate(chroma_mean):
            features[f'chroma_{i}'] = val
            
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_audio_directory(dataset_path, output_csv="extracted_features.csv"):
    """
    Processes a directory containing subdirectories for classes (e.g., parkinson/, healthy/).
    Assumes binary classification: Parkinson (1) or Healthy (0).
    """
    data = []
    
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if not os.path.isdir(class_path):
            continue
            
        label = 1 if 'parkinson' in class_folder.lower() else 0
        
        for file in os.listdir(class_path):
            if file.endswith('.wav') or file.endswith('.mp3'):
                file_path = os.path.join(class_path, file)
                print(f"Processing {file_path}")
                features = extract_features_from_audio(file_path)
                if features:
                    features['target'] = label
                    data.append(features)
                    
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")
        return df
    return pd.DataFrame()
