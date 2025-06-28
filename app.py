import streamlit as st
import os
from PIL import Image
import torch
from models.model_loader import load_model, restore_image
from utils.img_uti import preprocess_image, postprocess_tensor

st.title(" ClearVision - Image Restoration")

uploaded_file = st.file_uploader("Upload a degraded image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    input_path = os.path.join("uploads", uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(input_path, caption="Uploaded Image", use_column_width=True)


    model = load_model()


    input_tensor = preprocess_image(input_path)


    with torch.no_grad():
        restored_tensor = restore_image(model, input_tensor)


    restored_image = postprocess_tensor(restored_tensor)

   # restored_image = restored_image.crop((2, 2, restored_image.width - 2, restored_image.height - 2))
    st.image(restored_image, caption="Restored Image", use_column_width=True)


    restored_path = os.path.join("outputs", "restored_" + uploaded_file.name)
    restored_image.save(restored_path)
    st.download_button("Download Restored Image", open(restored_path, "rb"), file_name="restored.png")
