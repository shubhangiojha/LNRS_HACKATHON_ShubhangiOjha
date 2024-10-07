import cv2
import pytesseract
import streamlit as st
from PIL import Image
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def load_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    return np.array(img)

def preprocess_image(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh

def extract_text(image):
    if image is None:
        st.warning("No image provided for text extraction!")
        return ""
    processed_img = preprocess_image(image)
    text = pytesseract.image_to_string(processed_img)
    return text

st.title("Handwritten Notes Recognition")
st.write("Upload an image to extract handwritten text using Tesseract-OCR.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = load_image(uploaded_file)
    
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Extract Text"):
        extracted_text = extract_text(img)
        st.text_area("Extracted Text", extracted_text, height=200)
