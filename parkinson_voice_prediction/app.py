import os
import io
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from backend.prediction import predict_audio

# Use Agg backend for matplotlib to avoid thread/GUI issues
matplotlib.use('Agg')

app = Flask(__name__, static_folder='frontend', static_url_path='')

UPLOAD_FOLDER = 'dataset/temp_uploads'
PLOT_FOLDER = 'frontend/static/plots'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def generate_audio_plots(file_path):
    """Generates waveform and spectrogram for the uploaded audio."""
    base_name = os.path.basename(file_path).split('.')[0]
    waveform_path = os.path.join(PLOT_FOLDER, f"waveform_{base_name}.png")
    spectrogram_path = os.path.join(PLOT_FOLDER, f"spectrogram_{base_name}.png")
    
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Waveform
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.8, color="#1f77b4")
        plt.title('Waveform')
        plt.tight_layout()
        plt.savefig(waveform_path)
        plt.close()
        
        # Spectrogram
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar(img, format="%+2.0f dB")
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.savefig(spectrogram_path)
        plt.close()
        
        import numpy as np # need numpy here for np.abs and np.max
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        return None, None
        
    return waveform_path, spectrogram_path

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
        
    if file and (file.filename.endswith('.wav') or file.filename.endswith('.mp3') or file.filename.endswith('.webm')):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Ensure numpy is imported right because generate_audio_plots needs it
        import numpy as np
        
        # Hack to re-define with numpy imported for audio plots
        def make_plots(fp):
            base_name = os.path.basename(fp).split('.')[0]
            wp = os.path.join(PLOT_FOLDER, f"waveform_{base_name}.png")
            sp = os.path.join(PLOT_FOLDER, f"spectrogram_{base_name}.png")
            try:
                y, sr = librosa.load(fp, sr=None)
                plt.figure(figsize=(10, 4))
                librosa.display.waveshow(y, sr=sr, alpha=0.8, color="#1f77b4")
                plt.title('Waveform')
                plt.tight_layout()
                plt.savefig(wp)
                plt.close()
                plt.figure(figsize=(10, 4))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
                plt.colorbar(format="%+2.0f dB")
                plt.title('Spectrogram')
                plt.tight_layout()
                plt.savefig(sp)
                plt.close()
                return wp, sp
            except Exception as e:
                print(e)
                return None, None
                
        waveform_path, spectrogram_path = make_plots(file_path)
        
        # Predict using backend pipeline
        # Pass PLOT_FOLDER to save shap plot there
        result = predict_audio(file_path, output_shap_dir=PLOT_FOLDER)
        
        if 'error' in result:
            return jsonify(result)
            
        # Transform paths to be served by web browser (relative to static root)
        def format_path(p):
            if p and p.startswith('frontend/'):
                return p.replace('frontend/', '', 1)
            return p
            
        response = {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'waveform': format_path(waveform_path),
            'spectrogram': format_path(spectrogram_path),
            'shap': format_path(result.get('shap_plot_path'))
        }
        
        return jsonify(response)
        
    return jsonify({'error': 'Invalid file format. Please upload .wav, .mp3, or record live (.webm).'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
