"""
SD Turbo Image Generation Server
GTX 1650 Ti fix: UNet runs in float16, VAE decode in float32.

Install:
    pip install "diffusers==0.27.2" "transformers==4.40.0" "huggingface_hub==0.23.4" "accelerate==0.27.2" torch torchvision pillow flask flask-cors numpy
    pip uninstall xformers peft -y

Run:
    python sdxl_turbo_server.py
"""

import base64
import io
import sys
import time
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

print("Loading SD Turbo...")

try:
    import torch
    from PIL import Image
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    # Load UNet/text encoder in float16 for speed
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    # Keep VAE in float32 — fixes blank image bug on GTX 1650 Ti
    pipe.vae = pipe.vae.to(torch.float32)

    def generate_image(prompt):
        with torch.no_grad():
            # Run pipeline up to latents (float16, fast)
            latents = pipe(
                prompt=prompt,
                num_inference_steps=4,
                guidance_scale=0.0,
                width=512,
                height=512,
                output_type="latent",
            ).images

            # Decode in float32 (fixes blank output on Turing GPUs)
            latents_f32 = latents.to(torch.float32) / pipe.vae.config.scaling_factor
            decoded = pipe.vae.decode(latents_f32).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            img = Image.fromarray(
                (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
        return img

    # Warmup
    print("Warming up...")
    test = generate_image("a red apple on a table")
    arr = np.array(test)
    print(f"Warmup done — std={arr.std():.1f} ({'REAL' if arr.std()>5 else 'BLANK'})")
    if arr.std() < 5:
        print("ERROR: still blank after fix — contact support")
        sys.exit(1)

    print("SD Turbo ready at http://localhost:5556")

except Exception as e:
    print(f"Failed to load: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' field"}), 400

        prompt = data["prompt"]
        t0 = time.time()
        img = generate_image(prompt)
        elapsed = time.time() - t0

        arr = np.array(img)
        print(f"Generated in {elapsed:.1f}s — std={arr.std():.1f} — {prompt[:60]}...")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return jsonify({"image": b64})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "sd-turbo"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5556, debug=False)
