import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import re

import cv2
import numpy as np

def preprocess_image(pil_image):
    img = np.array(pil_image.convert("L"))  # Convert to grayscale
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(img)

st.set_page_config(page_title="Extract Report", layout="centered")
st.title("🧪 Automatic Extraction of Values from Blood Test Reports")

st.markdown("Upload or photograph a lab report: the app will extract **Test Name**, **Value**, and **Unit of Measure**.")

uploaded_file = st.file_uploader("📷 Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])

def extract_values(text):
    lines = text.split("\n")
    results = []
    pattern = r"([A-Za-z0-9 #\(\)/%µ\^\-]+?)\s+([\d.,]+)\s*([a-zA-Z/µ^%³]+)?"
    
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            test, value, unit = match.groups()
            try:
                value_float = float(value.replace(",", "."))
                results.append({
                    "Test": test.strip(),
                    "Value": value_float,
                    "Unit": unit or ""
                })
            except:
                continue
    return results

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Report", use_container_width=True)

    image = preprocess_image(Image.open(uploaded_file))

    
    with st.spinner("📖 Extracting values..."):
        text = pytesseract.image_to_string(image, lang="eng+ita")
        data = extract_values(text)

    if data:
        df = pd.DataFrame(data)
        st.success("✅ Values extracted successfully")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download as CSV", data=csv, file_name="lab_report_values.csv", mime="text/csv")
    else:
        st.warning("No values were recognized.")
