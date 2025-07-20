#in terminal: streamlit run streamlit_app.py

import streamlit as st
from PIL import Image
import io

from app_temp import custom_prompt
from ollama_utils import OllamaClient

from ollama_ocr import OCRProcessor
import tempfile

import cv2
import numpy as np
import pandas as pd

import re

#help(OCRProcessor.process_image)

def preprocess_image(pil_image):
    img = np.array(pil_image.convert("L"))  # Convert to grayscale
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(img)


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

# Page configuration

st.set_page_config(page_title="Extract Report", layout="centered")
st.title("🧪 Automatic Extraction of Values from Blood Test Reports")

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
        margin: 1rem 0;
    }
    .error-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        color: #721c24;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    "Upload or photograph a lab report: the app will extract **Test Name**, **Value**, and **Unit of Measure**.")

uploaded_file = st.file_uploader("📷 Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])


if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Report", use_container_width=True)# use_container_width=True)
        image = preprocess_image(Image.open(uploaded_file))
        st.image(image, caption="Uploaded Report", use_container_width=True)  # use_container_width=True)


    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        image = None

    if uploaded_file is not None:

        prompt_choice = st.radio("Choose a prompt type:",
                                 ["Ollama-OCR Default Prompt", "Lab Prompt", "Custom Prompt"],
                                 index=0 #Default selected
                                 )

        if prompt_choice == "Custom Prompt":

            custom_prompt = st.text_area(
                "Custom Prompt (optional)",
                placeholder="Enter a custom prompt to guide the analysis...",
                help="Leave empty to use the default prompt for general text extraction and analysis."
            )

        elif prompt_choice == "Lab Prompt":
            custom_prompt = """Using default prompt: Extract all blood count values content from this image in en **exactly as it appears**, without modification, summarization, or omission.
            Format the output in markdown:
            - output always test name, value and unit (if present)
            - Use headers (#, ##, ###) **only if they appear in the image**
            - Preserve original lists (-, *, numbered lists) as they are
            - Maintain all text formatting (bold, italics, underlines) exactly as seen
            - **Do not add, interpret, or restructure any content**
                """

        if st.button("Analyze Image"):

            with st.spinner("📖 Extracting values..."):

                try:

                    # Save uploaded PIL image to a temp file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        image.save(tmp.name)
                        temp_path = tmp.name

                    ocr = OCRProcessor(model_name="llama3.2-vision:11b")  # llama3.2-vision:11b #gemma3:4b

                    if prompt_choice != "Ollama-OCR Default Prompt":
                        prompt = custom_prompt

                    else:
                        prompt = None


                    result = ocr.process_image(
                            image_path=temp_path,
                            preprocess=True,
                            format_type="markdown",  # Options: markdown, text, json, structured, key_value
                            # language="en",
                            custom_prompt=prompt
                        )

                    print('#####################')
                    print(result)

                    st.subheader("📝 Results:")
                    st.text_area("Recognized Text", result, height=200)

                    data = extract_values(result)

                    if data:
                        df = pd.DataFrame(data)
                        st.success("✅ Values extracted successfully")
                        st.dataframe(df)
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download as CSV", data=csv, file_name="lab_report_values.csv", mime="text/csv")
                    else:
                        st.warning("No values were recognized.")

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Powered by Breeflee and Ollama*")