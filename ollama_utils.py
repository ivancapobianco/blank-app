import base64
import requests
from typing import Optional
import io
from PIL import Image

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
            "model": "gemma3:4b",
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
