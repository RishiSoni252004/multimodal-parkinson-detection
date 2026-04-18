# Parkinson's Disease Prediction Using Voice Analysis & Spiral Drawings

This project builds a **multi-modal machine learning system** that predicts whether a person may have Parkinson's disease based on **voice recordings** and **spiral drawing images**.

## Features
- **Voice Analysis Pipeline**: Random Forest, SVM, and XGBoost models trained on acoustic biomarkers.
- **Spiral Drawing Analysis**: ResNet-18 CNN fine-tuned on spiral drawings to detect motor impairment.
- **Voice Feature Extraction**: Librosa-based extraction of MFCC, Pitch, Spectral Centroid, ZCR, and Chroma Features.
- **Deep Learning**: Wav2Vec 2.0 integration for raw audio embeddings, plus PyTorch CNN for image classification.
- **Explainable AI**: SHAP for feature importance visualization.
- **Web Interface**: Streamlit dashboard supporting audio upload/recording, clinical feature input, and spiral image upload.
- **Data Cleaning**: Automated dataset cleaning for both audio (silence trimming) and images (resizing, normalization).

## Project Structure
- `dataset/` — Training data: `.wav` audio files, `.csv` voice features, and spiral drawing images.
  - `dataset/healthy/` & `dataset/parkinson/` — Voice recordings by class.
  - `dataset/spiral/healthy/` & `dataset/spiral/parkinson/` — Spiral drawing images by class.
- `models/` — Trained models, scalers, and feature metadata.
- `feature_extraction/` — Scripts to extract features from audio.
- `training/` — Model training logic for both voice and spiral pipelines.
- `prediction/` — Prediction engine (voice + spiral).
- `preprocessing/` — Data cleaning and preprocessing utilities.
- `backend/` and `frontend/` — Web interface components.
- `train_model.py` — Train voice analysis models.
- `training/train_spiral.py` — Train spiral drawing CNN model.
- `clean_datasets.py` — Clean and standardize both datasets.
- `app.py` — Streamlit web dashboard entry point.
- `start_all.sh` / `stop_all.sh` — One-command start/stop scripts.

## Setup and Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Clean the Datasets** (optional but recommended):
   ```bash
   python clean_datasets.py
   ```

3. **Train the Voice Model**:
   ```bash
   python train_model.py
   ```
   The best performing model will be saved in `models/best_model.pkl`.

4. **Train the Spiral Drawing Model**:
   Place spiral images into `dataset/spiral/healthy/` and `dataset/spiral/parkinson/`, then run:
   ```bash
   python training/train_spiral.py
   ```
   The CNN weights will be saved to `models/spiral_model.pth`.

5. **Run the Web App**:
   ```bash
   ./start_all.sh
   ```
   Access the dashboard at `http://localhost:8501`.

6. **Stop the Web App**:
   ```bash
   ./stop_all.sh
   ```

## Input Methods
| Method | Description |
|--------|-------------|
| 🎧 Audio Upload | Upload or record a voice sample for automated feature extraction and prediction. |
| 📊 Clinical Features | Paste pre-extracted acoustic feature values for direct model inference. |
| ✍️ Spiral Drawing Upload | Upload a spiral drawing image for CNN-based visual analysis. |

## Example Output
**Prediction**: Parkinson's Detected  
**Confidence**: 91.2%
