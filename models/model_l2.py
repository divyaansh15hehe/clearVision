import torch
import os
from .unet_generator import UNetGenerator
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model():
    model = UNetGenerator()
    path = os.path.join(BASE_DIR, "unet_model.pth")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def restore_image(model, tensor):
    with torch.no_grad():
        return model(tensor)