import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.prediction import predict_audio

def main():
    parser = argparse.ArgumentParser(description="Predict Parkinson's Disease from an audio file.")
    parser.add_argument("audio_file", help="Path to the .wav or .mp3 file to analyze.")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found.")
        return
        
    print(f"Analyzing '{args.audio_file}'...")
    
    # By default, save plots in the local directory for CLI usage
    result = predict_audio(args.audio_file, output_shap_dir=".")
    
    if "error" in result:
        print(f"Error during prediction: {result['error']}")
        return
        
    print("-" * 30)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    if result.get('shap_plot_path') and os.path.exists(result['shap_plot_path']):
        print(f"SHAP feature importance plot saved to: {result['shap_plot_path']}")
    print("-" * 30)

if __name__ == "__main__":
    main()
