import streamlit as st
from PIL import Image
import torch
import numpy as np
import io

# Set the title for the app
st.title("Aplikasi Deteksi Alat Olahraga")
st.markdown("Unggah gambar untuk mendeteksi alat olahraga menggunakan model YOLOv5.")

# Load YOLO model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Load custom YOLO model
    return model

model = load_model()

# File uploader for image input
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Perform detection
    st.write("Memproses gambar...")
    results = model(image)

    # Draw bounding boxes on the image
    detected_image = np.array(results.render()[0])

    # Display results
    st.image(detected_image, caption="Hasil Deteksi", use_column_width=True)
    st.write("Deteksi selesai. Berikut adalah hasil deteksinya:")
    st.dataframe(results.pandas().xyxy[0])  # Show detection details in a table

st.markdown("---")
st.markdown("Aplikasi ini menggunakan YOLOv5 untuk mendeteksi alat olahraga. Pastikan model `best.pt` sudah dikonfigurasi dengan benar.")
