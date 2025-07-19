import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import pytesseract
import layoutparser as lp
import tempfile

st.set_page_config(page_title="OCR Table Extractor", layout="centered")
st.title("📄 OCR Table Extractor (Tesseract + LayoutParser)")

uploaded_file = st.file_uploader("📷 Upload lab report (JPG/PNG)", type=["jpg", "jpeg", "png"])

def preprocess_image(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

@st.cache_resource
def load_layout_model():
    return lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    )

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Performing OCR..."):
        processed = preprocess_image(image)

        # Display raw OCR result
        raw_text = pytesseract.image_to_string(processed)
        st.subheader("📝 OCR Text Output")
        st.text(raw_text)

        # Detect layout blocks (tables, etc.)
        st.subheader("📊 Detected Tables (experimental)")
        layout_model = load_layout_model()
        layout = layout_model.detect(processed)

        tables = [b for b in layout if b.type == "Table"]

        if not tables:
            st.warning("⚠️ No tables detected.")
        else:
            for i, table in enumerate(tables):
                x1, y1, x2, y2 = map(int, table.coordinates)
                cropped = processed[y1:y2, x1:x2]
                table_text = pytesseract.image_to_string(cropped)

                rows = [r.split() for r in table_text.split("\n") if len(r.strip()) > 0]
                df = pd.DataFrame(rows)

                st.write(f"📄 Table {i+1}")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(f"⬇️ Download Table {i+1} as CSV", data=csv, file_name=f"table_{i+1}.csv")