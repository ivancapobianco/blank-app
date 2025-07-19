import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import numpy as np
import cv2

st.set_page_config(page_title="Extract Lab Report", layout="centered")
st.title("🧪 Smart Lab Value Extraction")

uploaded_file = st.file_uploader("📷 Upload lab report image", type=["jpg", "jpeg", "png"])

def preprocess_image(pil_image):
    img = np.array(pil_image.convert("L"))
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

def extract_table_data(image):
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME, config=custom_config)
    data = data[data.conf != -1]  # remove garbage lines
    data = data[data.conf > 60]  # keep only confident words

    lines = []
    current_line_num = -1
    current_line = []

    for _, row in data.iterrows():
        if row['line_num'] != current_line_num:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [row['text']]
            current_line_num = row['line_num']
        else:
            current_line.append(row['text'])

    if current_line:
        lines.append(" ".join(current_line))

    return lines

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
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Uploaded Report", use_container_width=True)

    with st.spinner("🔎 Processing and extracting..."):
        preprocessed = preprocess_image(pil_image)
        text_lines = extract_table_data(preprocessed)
        parsed_values = parse_lines_to_values(text_lines)

    if parsed_values:
        df = pd.DataFrame(parsed_values)
        st.success("✅ Values extracted with confidence")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="lab_report.csv", mime="text/csv")
    else:
        st.warning("⚠️ No values confidently recognized.")
