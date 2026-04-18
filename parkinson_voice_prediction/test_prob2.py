import torch
from prediction.predictor import Predictor
p = Predictor()
try:
    print(p.predict_from_audio("dataset/healthy/healthy_1.wav"))
except Exception as e:
    print(e)
