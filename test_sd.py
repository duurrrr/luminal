"""
Diagnoses blank image issue and tests the fix.
Run: python test_sd.py
"""

import sys
import torch
import numpy as np

try:
    from PIL import Image
    from diffusers import StableDiffusionPipeline, AutoencoderKL
except ImportError as e:
    print(f"Missing: {e}")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
print(f"Device: {device} | PyTorch: {torch.__version__}")

PROMPT = "a bright red apple on a wooden table"

print("\nLoading sd-turbo...")
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)

# ── Test 1: default (what you have now) ───────────────────────────────────────
print("\nTest 1: default settings...")
from diffusers import EulerDiscreteScheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)
with torch.no_grad():
    out = pipe(PROMPT, num_inference_steps=1, guidance_scale=0.0, width=512, height=512)
img = out.images[0]
arr = np.array(img)
print(f"  mean={arr.mean():.1f}  std={arr.std():.1f}  -> {'REAL' if arr.std()>5 else 'BLANK'}")
img.save("test1_default.png")

# ── Test 2: force VAE into float32 for decode only ────────────────────────────
print("\nTest 2: VAE decode in float32...")
pipe.vae = pipe.vae.to(torch.float32)
with torch.no_grad():
    # run unet in fp16, decode in fp32
    latents = pipe(PROMPT, num_inference_steps=1, guidance_scale=0.0,
                   width=512, height=512, output_type="latent").images
    latents_f32 = latents.to(torch.float32) / pipe.vae.config.scaling_factor
    decoded = pipe.vae.decode(latents_f32).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    img2 = Image.fromarray((decoded[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
pipe.vae = pipe.vae.to(dtype)  # restore
arr2 = np.array(img2)
print(f"  mean={arr2.mean():.1f}  std={arr2.std():.1f}  -> {'REAL' if arr2.std()>5 else 'BLANK'}")
img2.save("test2_vae_fp32.png")

# ── Test 3: full pipe in float32 ──────────────────────────────────────────────
print("\nTest 3: entire pipeline in float32...")
pipe32 = pipe.to(torch.float32)
with torch.no_grad():
    out3 = pipe32(PROMPT, num_inference_steps=1, guidance_scale=0.0, width=512, height=512)
img3 = out3.images[0]
arr3 = np.array(img3)
print(f"  mean={arr3.mean():.1f}  std={arr3.std():.1f}  -> {'REAL' if arr3.std()>5 else 'BLANK'}")
img3.save("test3_fp32.png")

print("\nDone — open test1/2/3 PNG files and check which are real images.")
print("Paste results here and I'll lock in the fix.")
