import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import re
import cv2

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

st.set_page_config(page_title="Extract Report", layout="centered")
st.title("🧪 Smart Blood Report Extraction (with DocTR)!!")

st.markdown("Upload or photograph a lab report: the app will extract **Test Name**, **Value**, and **Unit of Measure** using deep learning-based OCR.")

uploaded_file = st.file_uploader("📷 Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    return ocr_predictor(pretrained=True)

def parse_lines_to_values(lines):
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

    with st.spinner("🔍 Running layout-aware OCR..."):
        #doc = DocumentFile.from_images(image)
        doc = DocumentFile.from_images([image])


        model = load_model()
        result = model(doc)

        # Extract lines from prediction
        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    text = " ".join(word.value for word in line.words)
                    lines.append(text)

        parsed = parse_lines_to_values(lines)

    if parsed:
        df = pd.DataFrame(parsed)
        st.success("✅ Values extracted successfully")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="lab_report_values.csv", mime="text/csv")
    else:
        st.warning("⚠️ No values recognized.")
