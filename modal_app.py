# modal_app.py

from modal import App, Image, Secret, method
from fastapi import FastAPI
from fastapi.responses import FileResponse
import io
import torch
import torchaudio

# --- 1. Modal Image Definition ---
# This defines the environment where your app runs.

# We start with a base image that already has PyTorch and CUDA support
# We then install the TTS library and the API framework
image = (
    Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04") # Use a robust CUDA base image
    .apt_install("git", "ffmpeg", "libsndfile1") # Install necessary system dependencies
    .pip_install(
        "chatterbox-tts",
        "fastapi",
        "uvicorn",
        "torchaudio",
        "pydantic" # For API validation
    )
)

app = App(
    "chatterbox-tts-api", 
    image=image, 
    secrets=[Secret.from_name("my-modal-secret")] # Placeholder for potential future API key
)

# --- 2. Modal Class and Initialization ---

@app.cls(gpu="l4") # *** Crucial: Specifies the GPU to use ***
class ChatterboxService:
    def __enter__(self):
        """
        Modal's instance initialization method. 
        This loads the model once when the container starts (cold-start).
        """
        print("Loading Chatterbox TTS model...")
        try:
            # Import inside the class to ensure it's loaded in the Modal environment
            from chatterbox.tts import ChatterboxTTS
            
            # Load the model and map it to the GPU (cuda)
            self.model = ChatterboxTTS.from_pretrained(device="cuda")
            self.sample_rate = self.model.sr
            print("Model loaded successfully on GPU.")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @method()
    def generate_audio(self, text: str) -> bytes:
        """
        The core method to generate audio.
        """
        if not text:
            raise ValueError("Text cannot be empty.")
            
        print(f"Generating audio for text: '{text[:50]}...'")
        
        # 1. Generate the audio tensor
        wav_tensor = self.model.generate(text)
        
        # 2. Convert the tensor to a memory buffer (MP3/WAV file)
        buffer = io.BytesIO()
        
        # Ensure the tensor is on the CPU and is 1D (mono audio)
        wav_cpu = wav_tensor.cpu().float()
        
        # Save the audio to the buffer as WAV (higher quality, widely supported)
        torchaudio.save(
            buffer,
            wav_cpu.unsqueeze(0), # Add a channel dimension (1, length)
            self.sample_rate,
            format="wav" # WAV is simple and lossless
        )
        
        buffer.seek(0)
        return buffer.getvalue()


# --- 3. FastAPI API Setup for n8n ---

@app.web_endpoint(method="POST")
def generate_voiceover(service: ChatterboxService, request: dict):
    """
    HTTP endpoint that n8n will call.
    It takes JSON, calls the generate_audio method, and returns a file.
    """
    
    # Precaution 1: Input Validation
    # Ensure the 'text' key is present and is a string
    text = request.get("text")
    if not isinstance(text, str) or not text.strip():
        return {"error": "Invalid or missing 'text' field in JSON payload."}, 400

    try:
        # Call the Modal service method to do the actual work
        audio_bytes = service.generate_audio(text)
        
        # Precaution 2: Security and API Key (If needed, add logic here)
        # E.g., if request.headers.get("x-api-key") != YOUR_KEY: ...
        
        # 3. Return the audio file directly to n8n
        return FileResponse(
            path=io.BytesIO(audio_bytes),
            media_type="audio/wav", # Matches the torchaudio save format
            filename="voiceover_scene.wav"
        )
        
    except Exception as e:
        print(f"Deployment error during generation: {e}")
        return {"error": f"Internal server error during audio generation: {str(e)}"}, 500
