import streamlit as st
from PIL import Image
import io
from ollama_utils import OllamaClient

from ollama_ocr import OCRProcessor
import tempfile

# Page configuration
st.set_page_config(
    page_title="Gemma OCR Assistant",
    page_icon="🔍",
    layout="wide"
)

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

# Initialize Ollama client
@st.cache_resource
def get_ollama_client():
    return OllamaClient()

client = get_ollama_client()

# Header
st.title("🔍 Gemma OCR Assistant")
st.markdown("""
This application uses Gemma 3B's multimodal capabilities to analyze images and extract text.
Upload an image and optionally provide a custom prompt to guide the analysis.
""")

# Sidebar with instructions
with st.sidebar:
    st.header("📝 Instructions")
    st.markdown("""
    1. Upload an image containing text
    2. (Optional) Provide a custom prompt
    3. Click 'Analyze Image' to process
    
    **Example Prompts:**
    - Extract and list all text from the image
    - Describe the layout and formatting of text
    - Analyze the context and meaning of the text
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Image Upload")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            image = None
    
    custom_prompt = st.text_area(
        "Custom Prompt (optional)",
        placeholder="Enter a custom prompt to guide the analysis...",
        help="Leave empty to use the default prompt for general text extraction and analysis."
    )

with col2:
    st.subheader("Analysis Results")
    if uploaded_file is not None and st.button("Analyze Image"):
        with st.spinner("Processing image..."):
            try:
                #result = client.analyze_image(image, custom_prompt)

                ##### TEST
                # Save uploaded PIL image to a temp file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    image.save(tmp.name)
                    temp_path = tmp.name


                ocr = OCRProcessor(model_name="gemma3:4b") #llama3.2-vision:11b #gemma3:4b

                result = ocr.process_image(
                    image_path=temp_path,
                    format_type="markdown",  # Options: markdown, text, json, structured, key_value
                    #language="eng",
                    custom_prompt="Extract all the text. Do not remove or add anything."
                )

                print('#####################')
                print(result)

                ####


                st.markdown("### Results:")
                st.write(result)
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Powered by Gemma 3B and Ollama*") 
