import base64
import requests
from typing import Optional
import io
from PIL import Image
import json


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def analyze_image(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """
        Analyze an image using Gemma 3B model via Ollama.
        
        Args:
            image: PIL Image object
            prompt: Optional custom prompt to guide the analysis
            
        Returns:
            str: Model's response
        """
        base64_image = self._encode_image(image)
        
        default_prompt = "Please analyze this image and extract any text you can see. " \
                        "Provide a detailed description of the content and context."
                        
        final_prompt = prompt if prompt else default_prompt
        
        payload = {
            "model": "gemma3:4b", #gemma3:4b #gemma:2b
            "prompt": final_prompt,
            "images": [base64_image],
            "stream": False
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"



    def analyze_image_json(self, image: Image.Image, prompt: Optional[str] = None) -> dict:
        base64_image = self._encode_image(image)

        default_prompt = (
             "Please analyze this image of a blood test report and extract ONLY the following lab tests:\n"
    "- Hemoglobin\n"
    "- Lymphocytes (or Lymphocites)\n"
    "- Neutrophils (or Neutrophile)\n"
    "- Mean Corpuscular Volume (MCV)\n\n"
    "Ignore all other values.\n\n"
    "Return the result as a JSON object with this format:\n"
    "{ \"text_blocks\": [ { \"test_name\": \"...\", \"value\": ..., \"unit\": \"...\" } ] }\n\n"
    "Ensure the values are numbers and not strings. Include the unit if present in the image."
        )
        final_prompt = prompt if prompt else default_prompt

        payload = {
            "model": "gemma3:4b",
            "prompt": final_prompt,
            "images": [base64_image],
            "stream": False
        }

        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            raw_output = response.json()["response"]

            # Try parsing the response as JSON
            try:
                return json.loads(raw_output)
            except json.JSONDecodeError:
                return {"error": "Model response is not valid JSON", "raw_response": raw_output}

        except requests.exceptions.RequestException as e:
            return {"error": f"Error communicating with Ollama: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}


    def analyze_img_ollama_ocr(self, image: Image.Image, prompt: Optional[str] = None) -> str:

        base64_image = self._encode_image(image)

        ocr = OCRProcessor(model_name="gemma3:4b")

        result = ocr.process_image(
            image_path=[base64_image],
            format_type="markdown", #Options: markdown, text, json, structured, key_value
        )

        return(result)