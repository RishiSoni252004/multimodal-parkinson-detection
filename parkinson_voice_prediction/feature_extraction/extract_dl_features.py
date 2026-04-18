import os
import numpy as np
import librosa
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

class VoiceDataProcessor:
    def __init__(self, target_sr=16000, duration=3.0):
        self.target_sr = target_sr
        self.duration = duration
        self.max_length = int(target_sr * duration)

    def preprocess_audio(self, file_path):
        """Loads, resamples, normalizes, trims, and pads/truncates audio."""
        try:
            # 1. Load and resample
            y, sr = librosa.load(file_path, sr=self.target_sr)
            
            # 2. Trim silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            if len(y_trimmed) > 0:
                y = y_trimmed
                
            # 3. Normalize audio (RMS normalization)
            rms = np.sqrt(np.mean(y**2))
            if rms > 0:
                y = y / rms
                
            # 4. Handle variable length (pad/truncate to 3 seconds)
            if len(y) > self.max_length:
                y = y[:self.max_length]
            elif len(y) < self.max_length:
                padding = self.max_length - len(y)
                y = np.pad(y, (0, padding), mode='constant')
                
            return y
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def extract_features(self, y):
        """Extracts MFCC, Deltas, and Mel Spectrogram and combines them."""
        # MFCC (40 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=self.target_sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # Delta and Delta-Delta
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mfcc_mean = np.mean(delta_mfcc.T, axis=0)
        
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        delta2_mfcc_mean = np.mean(delta2_mfcc.T, axis=0)
        
        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=self.target_sr)
        mel_mean = np.mean(mel.T, axis=0)
        
        # Combine all features into a single 1D feature vector
        feature_vector = np.concatenate([mfcc_mean, delta_mfcc_mean, delta2_mfcc_mean, mel_mean])
        return feature_vector

    # 4. Data Augmentation
    def add_noise(self, y, noise_factor=0.005):
        noise = np.random.normal(0, y.std(), y.size)
        return y + noise_factor * noise

    def pitch_shift(self, y, num_semitones=2):
        return librosa.effects.pitch_shift(y, sr=self.target_sr, n_steps=num_semitones)
        
    def time_shift(self, y, shift_max=0.2):
        shift = np.random.randint(self.target_sr * shift_max)
        direction = np.random.choice(['right', 'left'])
        if direction == 'right':
            return np.pad(y, (shift, 0), mode='constant')[:len(y)]
        else:
            return np.pad(y, (0, shift), mode='constant')[shift:]

    def process_directory(self, data_dir, augment=True):
        """Processes directory, extracts features and handles augmentations."""
        X = []
        labels = []
        
        class_names = ["healthy", "parkinson"]
        
        total_files = 0
        corrupted_files = 0
        
        print("\n--- Starting Data Processing ---")
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found.")
                continue
                
            files = [f for f in os.listdir(class_dir) if f.endswith(('.wav', '.mp3'))]
            for file in files:
                total_files += 1
                file_path = os.path.join(class_dir, file)
                
                # Preprocess
                y = self.preprocess_audio(file_path)
                
                # Skip corrupted/empty audio files
                if y is None or len(y) == 0:
                    corrupted_files += 1
                    continue
                    
                # Extract Original
                features = self.extract_features(y)
                X.append(features)
                labels.append(class_idx)
                
                if augment:
                    # Augment: Noise
                    y_noise = self.add_noise(y)
                    X.append(self.extract_features(y_noise))
                    labels.append(class_idx)
                    
                    # Augment: Pitch Shift
                    y_pitch = self.pitch_shift(y, num_semitones=2)
                    X.append(self.extract_features(y_pitch))
                    labels.append(class_idx)
                    
                    # Augment: Pitch Shift Down
                    y_pitch2 = self.pitch_shift(y, num_semitones=-2)
                    X.append(self.extract_features(y_pitch2))
                    labels.append(class_idx)
                    
                    # Augment: Time Shift
                    y_time = self.time_shift(y)
                    X.append(self.extract_features(y_time))
                    labels.append(class_idx)

        # Ensure dataset is not empty
        if len(X) == 0:
            raise ValueError("Dataset is empty after processing. Check data directory.")
            
        X = np.array(X)
        y = np.array(labels)
        
        print("\n--- Data Quality Summary ---")
        print(f"Total audio files found: {total_files}")
        print(f"Corrupted/Skipped files: {corrupted_files}")
        print(f"Dataset Size (features array): {X.shape}")
        print(f"Labels Size: {y.shape}")
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(["Healthy" if c==0 else "Parkinson" for c in unique], counts))
        print(f"Class Distribution (including augmentations): {class_dist}")
        
        return X, y
