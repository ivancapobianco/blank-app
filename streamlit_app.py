import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import layoutparser as lp
from paddleocr import PaddleOCR
import tempfile
import cv2

st.set_page_config(page_title="OCR Table Extractor", layout="centered")
st.title("📄 OCR Table Extractor (Text + Tables from Images)")

uploaded_file = st.file_uploader("📷 Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_ocr_model():
    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

@st.cache_resource
def load_layout_model():
    return lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )

def extract_tables(image: np.ndarray, ocr_result):
    layout_model = load_layout_model()
    layout = layout_model.detect(image)

    tables = [block for block in layout if block.type == "Table"]
    dfs = []

    for i, table_block in enumerate(tables):
        table_image = image[int(table_block.block.y_1):int(table_block.block.y_2),
                            int(table_block.block.x_1):int(table_block.block.x_2)]

        sub_result = ocr.ocr(table_image)
        texts = []
        for line in sub_result[0]:
            txt = line[1][0]
            texts.append(txt)

        # Naive parsing into rows
        rows = [line.split() for line in texts if len(line.strip()) > 0]
        df = pd.DataFrame(rows)
        dfs.append(df)

    return dfs

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Performing OCR..."):
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.save(tmp.name)
            image_cv = cv2.imread(tmp.name)
            ocr = load_ocr_model()
            result = ocr.ocr(tmp.name)

            # Extract text lines
            lines = [line[1][0] for line in result[0]]
            st.subheader("📝 OCR Text")
            for line in lines:
                st.text(line)

            # Try to extract tables
            st.subheader("📊 Extracted Tables")
            tables = extract_tables(image_cv, result)
            if tables:
                for i, df in enumerate(tables):
                    st.write(f"Table {i+1}")
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(f"Download Table {i+1} as CSV", csv, file_name=f"table_{i+1}.csv")
            else:
                st.warning("⚠️ No tables detected.")