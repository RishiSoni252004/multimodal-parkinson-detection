"""
main.py — Entry Point for Parkinson's Disease Detection System

Multi-modal early detection using:
  - Voice Analysis (Wav2Vec 2.0 + PyTorch classifier, or MFCC + VoiceFNN)
  - Spiral Drawing Analysis (ResNet-18 CNN)
  - Multi-Modal Fusion (weighted combination)

Usage:
  Training:
    python main.py --mode train
    python main.py --mode train --pipeline voice     # Only voice classifier
    python main.py --mode train --pipeline classical  # Only sklearn models
    python main.py --mode train --pipeline spiral     # Only spiral CNN

  Prediction:
    python main.py --mode predict --input sample.wav
    python main.py --mode predict --input sample.wav --wav2vec
    python main.py --mode predict --input spiral.png
    python main.py --mode predict --input sample.wav --drawing spiral.png  # Multi-modal fusion
"""

import os
import sys
import argparse

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_train(args):
    """Run the training pipeline."""
    pipeline = args.pipeline

    if pipeline in ("voice", "all"):
        print("\n" + "=" * 60)
        print(" TRAINING: Wav2Vec2 Voice Classifier")
        print("=" * 60)
        from models.voice_classifier import train_voice_classifier
        train_voice_classifier(
            data_dir="dataset/",
            max_epochs=30,
            batch_size=4,
            lr=1e-4,
            patience=5,
            checkpoint_dir="checkpoints",
        )

    if pipeline in ("classical", "all"):
        print("\n" + "=" * 60)
        print(" TRAINING: Classical ML Models")
        print("=" * 60)
        from training.train import train_and_compare_models
        train_and_compare_models()

    if pipeline in ("dl", "all"):
        print("\n" + "=" * 60)
        print(" TRAINING: Deep Learning Voice Model (MFCC)")
        print("=" * 60)
        from training.train_voice_dl import train_voice_dl_model
        train_voice_dl_model(data_dir="dataset/")

    if pipeline in ("spiral", "all"):
        print("\n" + "=" * 60)
        print(" TRAINING: Spiral Drawing CNN")
        print("=" * 60)
        from training.train_spiral import train_spiral_model
        train_spiral_model(data_dir="dataset/spiral", epochs=20)

    print("\n✅ Training complete.")


def run_predict(args):
    """Run prediction on input file(s)."""
    input_path = args.input
    drawing_path = args.drawing
    use_wav2vec = args.wav2vec

    if not input_path:
        print("Error: --input is required for prediction mode.")
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    from prediction.predictor import Predictor
    predictor = Predictor()

    # Determine input type
    ext = os.path.splitext(input_path)[1].lower()
    is_audio = ext in (".wav", ".mp3", ".webm", ".flac", ".ogg")
    is_image = ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

    if is_audio and drawing_path:
        # Multi-modal fusion
        if not os.path.exists(drawing_path):
            print(f"Error: Drawing file not found: {drawing_path}")
            sys.exit(1)

        print("\n🔮 Multi-Modal Fusion Prediction")
        print(f"   Audio: {input_path}")
        print(f"   Drawing: {drawing_path}")
        print("-" * 40)

        from fusion import fuse_from_files
        result = fuse_from_files(
            audio_path=input_path,
            image_path=drawing_path,
            use_wav2vec=use_wav2vec,
        )

        print(f"   Voice probability:   {result['voice_prob']:.4f}")
        print(f"   Drawing probability: {result['drawing_prob']:.4f}")
        print(f"   Fused probability:   {result['probability']:.4f}")
        print(f"   Prediction:          {result['label']}")
        print(f"   Confidence:          {result['confidence']} ({result['confidence_score']:.4f})")

    elif is_audio:
        # Voice-only prediction
        model_name = "Wav2Vec 2.0" if use_wav2vec else "DL Voice (MFCC)"
        print(f"\n🎙️ Voice Prediction ({model_name})")
        print(f"   Input: {input_path}")
        print("-" * 40)

        result, prob = predictor.predict_from_audio(input_path, use_wav2vec=use_wav2vec)
        print(f"   Prediction:  {result}")
        print(f"   Confidence:  {prob:.4f}")

    elif is_image:
        # Spiral drawing prediction
        print(f"\n✍️ Spiral Drawing Prediction")
        print(f"   Input: {input_path}")
        print("-" * 40)

        result, prob = predictor.predict_from_spiral_image(input_path)
        print(f"   Prediction:  {result}")
        print(f"   Confidence:  {prob:.4f}")

    else:
        print(f"Error: Unsupported file type: {ext}")
        print("Supported: .wav, .mp3, .webm (audio) or .png, .jpg, .jpeg (image)")
        sys.exit(1)

    print("-" * 40)
    print("✅ Prediction complete.")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Parkinson's Disease Detection — Multi-Modal System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train all models:      python main.py --mode train
  Train voice only:      python main.py --mode train --pipeline voice
  Predict from audio:    python main.py --mode predict --input sample.wav
  Predict with Wav2Vec:  python main.py --mode predict --input sample.wav --wav2vec
  Predict from image:    python main.py --mode predict --input spiral.png
  Multi-modal fusion:    python main.py --mode predict --input sample.wav --drawing spiral.png
        """
    )

    parser.add_argument(
        "--mode", required=True, choices=["train", "predict"],
        help="Operation mode: 'train' or 'predict'"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input file (audio or image) for prediction"
    )
    parser.add_argument(
        "--drawing", type=str, default=None,
        help="Path to spiral drawing image for multi-modal fusion"
    )
    parser.add_argument(
        "--wav2vec", action="store_true", default=False,
        help="Use Wav2Vec 2.0 model for voice prediction"
    )
    parser.add_argument(
        "--pipeline", type=str, default="all",
        choices=["all", "voice", "classical", "dl", "spiral"],
        help="Which training pipeline to run (default: all)"
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()
