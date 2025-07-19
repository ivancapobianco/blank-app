import streamlit as st
from PIL import Image
import tempfile
import pandas as pd

from mutabnet_wrapper import MuTAbNet

st.set_page_config(page_title="Lab Report Table Extractor", layout="centered")
st.title("📋 Lab Report Table Extraction")

uploaded = st.file_uploader("Upload a lab report image (JPG/PNG)", type=["jpg", "png", "jpeg"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Report", use_container_width=True)

    @st.cache_resource
    def load_model():
        return MuTAbNet.from_pretrained(device="cuda" if st.runtime.exists("gpu") else "cpu")

    model = load_model()

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        image.save(tmp.name)
        with st.spinner("🔍 Running table extraction..."):
            tables = model.predict(tmp.name)

    if tables:
        for i, df in enumerate(tables):
            st.subheader(f"Extracted Table {i+1}")
            st.dataframe(df)
            csv = df.to_csv(index=False).encode()
            st.download_button(f"Download Table {i+1} as CSV", data=csv, file_name=f"table_{i+1}.csv")
    else:
        st.warning("⚠️ No tables detected.")