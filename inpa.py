from diffusers.pipelines.stable_diffusion import safety_checker
import torch
from PIL import Image
from diffusers import FluxInpaintPipeline, FluxTransformer2DModel, FluxFillPipeline
from transformers import T5EncoderModel
from deep_translator import GoogleTranslator
from optimum.quanto import freeze, qfloat8, quantize
import argparse
import os

import torch
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from controlnet_aux import OpenposeDetector

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)

# --- ARGPARSE CONFIGURATION ---
parser = argparse.ArgumentParser(description='FLUX Inpaint/Fill Generator')

# Parametri principali
parser.add_argument('--mode', type=str, default='1', 
                    help='Modalità: "1" per Inpaint, "2" per Fill, "3" per ControlNet (default: 1)')
parser.add_argument('--path_image', type=str, required=True,
                    help='Path immagine di input')
parser.add_argument('--path_mask', type=str, required=True,
                    help='Path maschera')
parser.add_argument('--prompt', type=str, 
                    default='high quality image',
                    help='Prompt per la generazione')
parser.add_argument('--output', type=str, default='output',
                    help='Nome file output (senza estensione)')

# Parametri LoRA
parser.add_argument('--lora', type=str, default=None,
                    help='Nome file LoRA (opzionale)')
parser.add_argument('--scale_lora', type=float, default=0.8,
                    help='Peso LoRA (default: 0.8)')

# Parametri generazione
parser.add_argument('--steps', type=int, default=30,
                    help='Numero step inferenza (default: 30)')
parser.add_argument('--cfg', type=float, default=6.0,
                    help='Guidance scale (default: 6.0)')
parser.add_argument('--strength', type=float, default=0.80,
                    help='Strength per modifica (default: 0.80)')
parser.add_argument('--seed', type=int, default=24,
                    help='Seed per generatore (default: 24)')

args = parser.parse_args()

# Configurazione
DEVICE = "cuda"
mode_use = args.mode

# --- PREPARAZIONE IMMAGINI ---
image = Image.open(args.path_image).convert("RGB")
mask = Image.open(args.path_mask).convert("RGB")

target_size = 1024 
w, h = image.size
scale = min(target_size/w, target_size/h)
new_w, new_h = int(w * scale)//16*16, int(h * scale)//16*16
image = image.resize((new_w, new_h), Image.LANCZOS)
mask = mask.resize((new_w, new_h), Image.LANCZOS)

# --- CONFIGURAZIONE MODELLI PER MODALITÀ ---
control_image = None
pipe = None
result = None

if mode_use == '1':
    # INPAINTING
    base_repo = "black-forest-labs/FLUX.1-dev"
    model_path = "./model/fluxedUpFluxNSFW_51FP8.safetensors"
    pipeline_class = FluxInpaintPipeline
    mode_name = "Inpaint"
    dtype = torch.float16
    
elif mode_use == '2':
    # FILL
    base_repo = "black-forest-labs/FLUX.1-Fill-dev"
    model_path = "./model/agfluxFillNSFWFp8_agfluxFillNSFWV17Fp8.safetensors"
    pipeline_class = FluxFillPipeline
    mode_name = "Fill"
    dtype = torch.float16
    
elif mode_use == '3':
    # CONTROLNET
    checkpoint = "lllyasviel/control_v11p_sd15_openpose"
    processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    control_image = processor(image, hand_and_face=True)
    
    # Crea directory images se non esiste
    os.makedirs("./images", exist_ok=True)
    control_image.save("./images/control.png")
    
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    path_model = "./model/pornmasterProAsianv2_asianv2Fix-inpainting.safetensors"
    pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
        path_model, controlnet=controlnet, torch_dtype=torch.bfloat16, safety_checker=None
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    mode_name = "ControlNet"

print(f"=== Modalità: {mode_name} ===")

if mode_use in ['1', '2']:
    print(f"Repository base: {base_repo}")
    print(f"Modello custom: {model_path}")
    
    # --- CARICAMENTO TRANSFORMER PERSONALIZZATO ---
    print("\nCaricamento Transformer personalizzato...")
    transformer = FluxTransformer2DModel.from_single_file(
        model_path,
        torch_dtype=dtype
    )
    print("Quantizzazione Transformer...")
    quantize(transformer, weights=qfloat8)
    freeze(transformer)

    # --- CARICAMENTO TEXT ENCODER ---
    print("Caricamento Text Encoder...")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        base_repo,
        subfolder="text_encoder_2",
        torch_dtype=dtype
    )
    print("Quantizzazione Text Encoder...")
    quantize(text_encoder_2, weights=qfloat8)
    freeze(text_encoder_2)

    # --- CARICAMENTO PIPELINE ---
    print(f"Caricamento Pipeline {mode_name}...")
    if os.path.exists(f"./Lora/{args.lora}"):
        pipe = pipeline_class.from_pretrained(
            base_repo,
            torch_dtype=dtype
        )
    else:
        pipe = pipeline_class.from_pretrained(
            base_repo,
            transformer=None,
            text_encoder_2=None,
            torch_dtype=dtype
        )
# --- CARICAMENTO LORA (opzionale) ---
if args.lora and pipe is not None:
    lora_path = f"./Lora/{args.lora}"
    if os.path.exists(lora_path):
        print(f"Caricamento LoRA: {lora_path} (weight: {args.scale_lora})")
        pipe.load_lora_weights(lora_path, adapter_name='lora')
        pipe.set_adapters(['lora'], adapter_weights=[args.scale_lora])
    else:
        print(f"⚠️ Warning: LoRA non trovata in {lora_path}")
    
print("Sostituzione componenti quantizzati...")
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2
    
# CPU offload per gestione memoria
pipe.enable_model_cpu_offload()



# --- GENERAZIONE ---
prompt_ita = args.prompt

print(f"\nPrompt originale: '{prompt_ita[:100]}...'")
prompt = GoogleTranslator(source='it', target='en').translate(prompt_ita)
print(f"Prompt tradotto: '{prompt[:100]}...'")

generator = torch.Generator(DEVICE).manual_seed(args.seed)

print(f"\nGenerazione in corso ({mode_name})...")
print(f"Steps: {args.steps} | CFG: {args.cfg} | Strength: {args.strength} | Seed: {args.seed}")

try:
    if mode_use == '1':
        # FluxInpaintPipeline - con strength
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            height=new_h,
            width=new_w,
            num_inference_steps=args.steps,
            strength=args.strength,
            generator=generator,
            guidance_scale=args.cfg,
            max_sequence_length=512
        )
    elif mode_use == '2':
        # FluxFillPipeline
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            height=new_h,
            width=new_w,
            num_inference_steps=args.steps,
            generator=generator,
            guidance_scale=args.cfg,
            max_sequence_length=512
        )
    elif mode_use == '3':
        # ControlNet Pipeline
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            height=new_h,
            width=new_w,
            num_inference_steps=args.steps,
            generator=generator,
            guidance_scale=args.cfg,
            control_image=control_image
        )

except Exception as e:
    print(f"Errore durante la generazione: {e}")
    raise

# --- SALVATAGGIO ---
if result is not None:
    if hasattr(result, 'images') and result.images:
        output_img = result.images[0]
        output_filename = f"{args.output}.png"  # FIX: era arg.output invece di args.output
        output_img.save(output_filename)
        print(f"\n✓ Immagine salvata in {output_filename}")
    else:
        print("\n❌ Errore: nessuna immagine generata")
else:
    print("\n❌ Errore: result è None")