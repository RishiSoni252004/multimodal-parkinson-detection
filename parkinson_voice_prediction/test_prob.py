import torch
from models.spiral_model import SpiralModel
import glob

# Test on healthy images
h_imgs = glob.glob("dataset/spiral/healthy/*.png") + glob.glob("dataset/spiral/healthy/*.jpg")
# Test on parkinson images
p_imgs = glob.glob("dataset/spiral/parkinson/*.png") + glob.glob("dataset/spiral/parkinson/*.jpg")

model = SpiralModel("models/spiral_model.pth")
if h_imgs:
    print("Healthy img 1:", model.predict(h_imgs[0]))
if len(h_imgs) > 1:
    print("Healthy img 2:", model.predict(h_imgs[1]))

if p_imgs:
    print("Parkinson img 1:", model.predict(p_imgs[0]))
if len(p_imgs) > 1:
    print("Parkinson img 2:", model.predict(p_imgs[1]))

