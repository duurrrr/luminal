"""
BLIP Description Server
Loads Salesforce BLIP locally, exposes a simple HTTP endpoint
that accepts a base64 PNG and returns a text description.

Install once:
    pip install transformers torch torchvision pillow flask flask-cors

Run:
    python blip_server.py

Endpoint:
    POST http://localhost:5555/describe
    Body: { "image": "<base64 png string>" }
    Returns: { "description": "..." }
"""

import base64
import io
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)  # allow requests from the browser

print("Loading BLIP model... (this takes ~30 seconds the first time)")

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)

    print("BLIP model loaded. Server ready at http://localhost:5555")

except Exception as e:
    print(f"Failed to load BLIP: {e}")
    print("Make sure you ran: pip install transformers torch torchvision pillow flask flask-cors")
    sys.exit(1)


@app.route("/describe", methods=["POST"])
def describe():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        # Decode base64 image
        img_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Run BLIP with an open-ended prompt for detailed descriptions
        prompt = "a detailed description of"
        inputs = processor(image, prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=5,
                min_length=30,
            )

        description = processor.decode(out[0], skip_special_tokens=True)

        # Strip the prompt prefix if BLIP echoes it back
        if description.lower().startswith("a detailed description of"):
            description = description[len("a detailed description of"):].strip()

        print(f"Described: {description[:80]}...")
        return jsonify({"description": description})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "blip-image-captioning-large"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=False)
