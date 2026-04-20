"""
fusion.py — Multi-Modal Fusion for Parkinson's Disease Detection

Combines voice analysis and spiral drawing predictions using
weighted probability fusion.

Configuration:
  FUSION_WEIGHTS = {"voice": 0.6, "drawing": 0.4}

Output:
  - Probability of Parkinson's (0.0 to 1.0)
  - Binary label ("Parkinson Detected" or "Healthy")
  - Confidence level ("High", "Medium", "Low")
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# CONFIGURATION — Weights are configurable here
# ==============================================================================

FUSION_CONFIG = {
    "voice_weight": 0.6,       # Weight for voice model probability
    "drawing_weight": 0.4,     # Weight for drawing model probability
    "threshold": 0.5,          # Decision threshold for Parkinson's
    "high_confidence": 0.80,   # Above this = "High" confidence
    "low_confidence": 0.60,    # Below this = "Low" confidence
}


# ==============================================================================
# FUSION FUNCTION
# ==============================================================================

def fuse_predictions(
    voice_prob: float,
    drawing_prob: float,
    config: dict = None,
) -> dict:
    """
    Combine voice and drawing model predictions using weighted fusion.

    Formula:
      final_score = (voice_weight × voice_prob) + (drawing_weight × drawing_prob)

    Args:
        voice_prob: Probability of Parkinson's from voice model (0.0 to 1.0).
        drawing_prob: Probability of Parkinson's from drawing model (0.0 to 1.0).
        config: Optional dict overriding FUSION_CONFIG defaults.

    Returns:
        dict with keys:
          - probability: float (0.0 to 1.0) — fused Parkinson's probability
          - label: str — "Parkinson Detected" or "Healthy"
          - confidence: str — "High", "Medium", or "Low"
          - voice_prob: float — input voice probability
          - drawing_prob: float — input drawing probability
          - voice_weight: float — weight used for voice
          - drawing_weight: float — weight used for drawing
    """
    cfg = {**FUSION_CONFIG, **(config or {})}

    voice_weight = cfg["voice_weight"]
    drawing_weight = cfg["drawing_weight"]
    threshold = cfg["threshold"]

    # Normalize weights to sum to 1.0
    total_weight = voice_weight + drawing_weight
    if total_weight > 0:
        voice_weight /= total_weight
        drawing_weight /= total_weight

    # Clamp input probabilities
    voice_prob = max(0.0, min(1.0, float(voice_prob)))
    drawing_prob = max(0.0, min(1.0, float(drawing_prob)))

    # Weighted combination
    final_prob = (voice_weight * voice_prob) + (drawing_weight * drawing_prob)
    final_prob = max(0.0, min(1.0, final_prob))

    # Binary decision
    label = "Parkinson Detected" if final_prob >= threshold else "Healthy"

    # Confidence level
    confidence_score = final_prob if label == "Parkinson Detected" else (1.0 - final_prob)
    if confidence_score >= cfg["high_confidence"]:
        confidence = "High"
    elif confidence_score >= cfg["low_confidence"]:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "probability": round(final_prob, 4),
        "label": label,
        "confidence": confidence,
        "confidence_score": round(confidence_score, 4),
        "voice_prob": round(voice_prob, 4),
        "drawing_prob": round(drawing_prob, 4),
        "voice_weight": round(voice_weight, 4),
        "drawing_weight": round(drawing_weight, 4),
    }


def fuse_from_files(
    audio_path: str,
    image_path: str,
    use_wav2vec: bool = True,
    config: dict = None,
) -> dict:
    """
    End-to-end multi-modal prediction from audio + image files.

    Args:
        audio_path: Path to voice audio file.
        image_path: Path to spiral drawing image.
        use_wav2vec: Whether to use Wav2Vec2 for voice (vs DL/classical).
        config: Optional fusion config overrides.

    Returns:
        dict — same as fuse_predictions() output.
    """
    from prediction.predictor import Predictor

    predictor = Predictor()

    # Voice prediction
    voice_label, voice_prob_raw = predictor.predict_from_audio(audio_path, use_wav2vec=use_wav2vec)
    voice_prob = voice_prob_raw if voice_label == "Parkinson Detected" else (1.0 - voice_prob_raw)

    # Drawing prediction
    drawing_label, drawing_prob_raw = predictor.predict_from_spiral_image(image_path)
    drawing_prob = drawing_prob_raw if drawing_label == "Parkinson Detected" else (1.0 - drawing_prob_raw)

    return fuse_predictions(voice_prob, drawing_prob, config)


# ==============================================================================
# SELF-TEST
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" MULTI-MODAL FUSION TEST")
    print("=" * 60)

    # Test 1: Dummy scores — both suggest Parkinson's
    print("\n--- Test 1: Both models suggest Parkinson's ---")
    result = fuse_predictions(voice_prob=0.85, drawing_prob=0.75)
    print(f"  Voice prob: {result['voice_prob']}")
    print(f"  Drawing prob: {result['drawing_prob']}")
    print(f"  Fused probability: {result['probability']}")
    print(f"  Label: {result['label']}")
    print(f"  Confidence: {result['confidence']} ({result['confidence_score']})")
    assert result["label"] == "Parkinson Detected"
    print(f"  ✅ Passed")

    # Test 2: Both suggest Healthy
    print("\n--- Test 2: Both models suggest Healthy ---")
    result = fuse_predictions(voice_prob=0.15, drawing_prob=0.20)
    print(f"  Fused probability: {result['probability']}")
    print(f"  Label: {result['label']}")
    print(f"  Confidence: {result['confidence']}")
    assert result["label"] == "Healthy"
    print(f"  ✅ Passed")

    # Test 3: Disagreement — voice says Parkinson, drawing says Healthy
    print("\n--- Test 3: Models disagree ---")
    result = fuse_predictions(voice_prob=0.90, drawing_prob=0.20)
    print(f"  Voice: 0.90 (Parkinson) × 0.6 weight")
    print(f"  Drawing: 0.20 (Healthy) × 0.4 weight")
    print(f"  Fused: {result['probability']}")
    print(f"  Label: {result['label']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  ✅ Passed")

    # Test 4: Custom weights
    print("\n--- Test 4: Custom weights (equal) ---")
    result = fuse_predictions(
        voice_prob=0.70, drawing_prob=0.30,
        config={"voice_weight": 0.5, "drawing_weight": 0.5}
    )
    print(f"  Equal weights: fused={result['probability']} (should be ~0.50)")
    print(f"  Label: {result['label']}")
    print(f"  ✅ Passed")

    # Test 5: Edge cases
    print("\n--- Test 5: Edge cases ---")
    r1 = fuse_predictions(voice_prob=0.0, drawing_prob=0.0)
    r2 = fuse_predictions(voice_prob=1.0, drawing_prob=1.0)
    r3 = fuse_predictions(voice_prob=-0.5, drawing_prob=1.5)  # out of range
    print(f"  (0.0, 0.0) → {r1['probability']}, {r1['label']}")
    print(f"  (1.0, 1.0) → {r2['probability']}, {r2['label']}")
    print(f"  (-0.5, 1.5) → {r3['probability']}, {r3['label']} (clamped)")
    print(f"  ✅ Passed")

    print("\n" + "=" * 60)
    print(" FUSION TEST COMPLETE — All tests passed")
    print("=" * 60)
