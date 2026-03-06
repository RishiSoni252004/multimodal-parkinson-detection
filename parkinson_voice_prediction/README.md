# Parkinson's Disease Prediction Using Voice Analysis

This project aims to build a machine learning system that predicts whether a person may have Parkinson's disease based on voice recordings or extracted voice features.

## Features
- **Machine Learning Models**: Random Forest, Support Vector Machine (SVM), XGBoost.
- **Voice Feature Extraction**: Librosa to extract MFCC, Pitch, Spectral Centroid, ZCR, and Chroma Features.
- **Explainable AI**: SHAP for feature importance visualization.
- **Web Interface**: Flask + HTML/JS frontend to upload an audio file, view waveform/spectrogram, and get predictions.

## Project Structure
- `dataset/`: Place your training data here (either `.wav` files or a `.csv` with pre-extracted features).
- `models/`: Trained models and scalers are saved here.
- `feature_extraction/`: Scripts to extract features from audio.
- `training/`: Logic for loading data, training models, and evaluation.
- `backend/` and `frontend/`: Web interface components.
- `train_model.py`: Script to trigger training.
- `predict.py`: Script to run a single prediction via CLI.
- `app.py`: Web server entry point.

## Setup and Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Training the Model**
   Place your dataset in the `dataset/` folder.
   Run the training script:
   ```bash
   python train_model.py
   ```
   The best performing model will be saved in `models/best_model.pkl`.

3. **Running the Web App**
   ```bash
   python app.py
   ```
   Access the web interface at `http://localhost:5000` to upload a `.wav` file and see predictions.

## Example Output
**Prediction**: Parkinson's Detected
**Confidence**: 91.2%
