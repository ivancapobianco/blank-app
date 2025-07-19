import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="OCR Table Extractor", layout="centered")
st.title("📄 OCR Table Extractor (Tesseract)")

uploaded_file = st.file_uploader("📷 Upload lab report (JPG/PNG)", type=["jpg", "jpeg", "png"])

def extract_rows(text):
    lines = text.split('\n')
    rows = []
    pattern = r"([A-Za-z0-9 #\(\)/%µ\^\-]+?)\s+([\d.,]+)\s*([a-zA-Z/µ^%³]+)?"
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            test, value, unit = match.groups()
            try:
                value_float = float(value.replace(",", "."))
                rows.append({
                    "Test": test.strip(),
                    "Value": value_float,
                    "Unit": unit or ""
                })
            except:
                continue
    return rows

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Performing OCR..."):
        text = pytesseract.image_to_string(image)
        st.subheader("📝 OCR Text Output")
        st.text(text)

        parsed = extract_rows(text)

        if parsed:
            df = pd.DataFrame(parsed)
            st.success("✅ Extracted structured table")
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", data=csv, file_name="lab_report_values.csv")
        else:
            st.warning("⚠️ No structured values could be parsed.")