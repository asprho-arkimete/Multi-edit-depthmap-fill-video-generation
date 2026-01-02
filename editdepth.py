import webbrowser
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor
from PIL import Image
from diffusers import FluxTransformer2DModel
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize
from PIL import Image, ImageEnhance
from deep_translator import GoogleTranslator 
import tkinter as tk
from tkinter import ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import ImageTk, ImageOps
from pillow_heif import register_heif_opener

# Registra il supporto per HEIC/HEIF
register_heif_opener()
import os

def kontext(image_posa, image_ref, image_ref2, text):
    # ✅ STEP 0: Controllo iniziale
    if not image_posa or not image_ref:
        print("❌ Errore: carica almeno depth map e riferimento 1!")
        return
    
    # ✅ STEP 1: Carica l'immagine 3D e genera una depth map di qualità
    if os.path.exists(image_posa):  # Corretto: exists non exist
        source_3d = Image.open(image_posa).convert("RGB")
        processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
        depth_map = processor(source_3d)[0]
        
        # ✅ STEP 2: Migliora la depth map (aumenta contrasto)
        enhancer = ImageEnhance.Contrast(depth_map)
        depth_map = enhancer.enhance(1.8)
    else:
        print("❌ Errore: file depth map non trovato!")
        return

    # ✅ STEP 3: Carica foto di riferimento
    if os.path.exists(image_ref):
        reference_image = Image.open(image_ref).convert("RGB")
    else:
        print("❌ Errore: file riferimento 1 non trovato!")
        return
    
    # Carica riferimento 2 (opzionale)
    reference_image2 = None
    if image_ref2 and os.path.exists(image_ref2):  # Corretto: controllava due volte image_ref
        reference_image2 = Image.open(image_ref2).convert("RGB")
        print("✅ Riferimento 2 caricato")

    # ✅ STEP 4: Ridimensiona tutte le immagini alla STESSA dimensione
    def Resize(image):
        w, h = image.size
        rw, rh = 1024, 1024
        if w > 1024 or h > 1024:
            if w >= h:
                rh = (rw * h) // w
            else:
                rw = (rh * w) // h
            image = image.resize((rw, rh), Image.BICUBIC)
        return image

    # Ridimensiona le immagini
    control_image = Resize(depth_map)
    control_image.save('depth.png')
    
    image = Resize(reference_image)
    
    if reference_image2:  # Corretto: logica più semplice
        image2 = Resize(reference_image2)
    else:
        image2 = None

    # ✅ STEP 5: Crea collage - CASI POSSIBILI
    if reference_image2:  # Caso: depth + ref1 + ref2
        # Corretto: era max(control_image.size[1], image.size[1]), image2.size[1] - sintassi errata
        collante = Image.new('RGB', (
            control_image.size[0] + image.size[0] + image2.size[0], 
            max(control_image.size[1], image.size[1], image2.size[1])
        ))
        collante.paste(control_image, (0, 0))
        collante.paste(image, (control_image.size[0], 0))
        collante.paste(image2, (control_image.size[0] + image.size[0], 0))
        print(f"✅ Collage [Depth | Rif1 | Rif2]: {collante.size}")
        
    else:  # Caso: depth + ref1
        # Corretto: rimosso + finale in image.size[0]+
        collante = Image.new('RGB', (
            control_image.size[0] + image.size[0], 
            max(control_image.size[1], image.size[1])
        ))
        collante.paste(control_image, (0, 0))
        collante.paste(image, (control_image.size[0], 0))
        print(f"✅ Collage [Depth | Rif1]: {collante.size}")
    collante=Resize(collante)
    collante.save('collage_final.png')
    print("✅ Collage salvato: collage_final.png")
    
    # Setup del modello
    bfl_repo = "black-forest-labs/FLUX.1-Kontext-dev"
    dtype = torch.bfloat16

    transformer = FluxTransformer2DModel.from_pretrained(
        bfl_repo, subfolder="transformer", torch_dtype=dtype
    )
    quantize(transformer, weights=qfloat8)
    freeze(transformer)

    text_encoder_2 = T5EncoderModel.from_pretrained(
        bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype
    )
    quantize(text_encoder_2, weights=qfloat8)
    freeze(text_encoder_2)

    pipe = FluxKontextPipeline.from_pretrained(
        bfl_repo, torch_dtype=torch.bfloat16
    )
    # Carica il LoRA
    pipe.load_lora_weights("./LORA_flux_kontext_depth_reference_cotrol.safetensors", adapter_name="depth")
    pipe.set_adapters("depth", adapter_weights=1.5)

    pipe.transformer = transformer
    pipe.text_encoder_2 = text_encoder_2

    pipe.enable_model_cpu_offload()
    
    from transformers import T5Tokenizer
    # Carica il tokenizer di T5
    tokenizer = T5Tokenizer.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", subfolder="tokenizer_2")

    # Prompt con trigger word
    prompt = GoogleTranslator(source='it', target='en').translate(text)
    
    # Prompt adattivo
    final_prompt = f"From the provided reference images, create a unified, cohesive image such that {prompt}. Maintain the identity and characteristics of each subject while adjusting their proportions, scale, and positioning to create a harmonious, naturally balanced composition. Blend and integrate all elements seamlessly with consistent lighting, perspective, and style.the final result should look like a single naturally captured scene where all subjects are properly sized and positioned relative to each other, not assembled from multiple sources."
    tokens = tokenizer.encode(final_prompt)
    print(f"📝 Token: {len(tokens)} | Caratteri: {len(prompt)} | Rapporto: {len(prompt) / len(tokens):.2f}")

    print("🎨 Generazione in corso...")
    image_out = pipe(
        image=collante,
        prompt=final_prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
        width=depth_map.width,
        height=depth_map.height,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]

    image_out.save("result_final.png")
    print("✅ Immagine salvata: result_final.png")

# ===== INTERFACCIA GRAFICA =====

window = TkinterDnD.Tk()
window.config(bg='gray')
window.title("Edit image 3d")
window.geometry("1580x720")
window.resizable(False, False)

frames_Canvas = tk.Frame(window)
frames_Canvas.grid(row=0, column=0)

# Variabili globali per i path
path1 = None
path2 = None
path3 = None

def load(event, frame, path_var):
    global path1, path2, path3
    
    path = event.data
    # Rimuovi caratteri extra che potrebbero essere presenti nel path
    if path.startswith('{'):
        path = path[1:-1]
    
    rw, rh = 512, 512
    img = Image.open(path)
    # CORREGGI L'ORIENTAMENTO BASANDOTI SUI METADATI EXIF
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    
    if w > rw or h > rh:
        if w >= h:
            # Mantieni aspect ratio
            new_h = (rw * h) // w
            img = img.resize((rw, new_h))
        else:
            new_w = (rh * w) // h
            img = img.resize((new_w, rh))
    
    # Converti in PhotoImage per tkinter
    photo = ImageTk.PhotoImage(img)
    frame.image = photo  # Mantieni un riferimento per evitare garbage collection
    frame.delete("all")  # Pulisci il canvas
    frame.create_image(256, 256, image=photo)
    frame.update_idletasks()
    
    # Aggiorna la variabile globale corretta
    if path_var == 1:
        path1 = path
        print(f"✅ Path1 (Posa) caricato: {path}")
    elif path_var == 2:
        path2 = path
        print(f"✅ Path2 (Riferimento 1) caricato: {path}")
    elif path_var == 3:  # Aggiunto caso per path3
        path3 = path
        print(f"✅ Path3 (Riferimento 2) caricato: {path}")

# Frame 1 - Depth/Posa
frame1 = tk.Canvas(frames_Canvas, width=512, height=512, bg='red')   
frame1.grid(row=0, column=0, padx=5)
frame1.create_text(256, 256, text='Carica foto posa', font=('Arial', 20), fill='white')
frame1.drop_target_register(DND_FILES)
frame1.dnd_bind('<<Drop>>', lambda e: load(e, frame1, 1))

# Frame 2 - Riferimento 1
frame2 = tk.Canvas(frames_Canvas, width=512, height=512, bg='red')   
frame2.grid(row=0, column=1, padx=5)
frame2.create_text(256, 256, text='Carica foto riferimento 1', font=('Arial', 20), fill='white')
frame2.drop_target_register(DND_FILES)
frame2.dnd_bind('<<Drop>>', lambda e: load(e, frame2, 2))

# Frame 3 - Riferimento 2
frame3 = tk.Canvas(frames_Canvas, width=512, height=512, bg='red')   
frame3.grid(row=0, column=2, padx=5)
frame3.create_text(256, 256, text='Carica foto riferimento 2', font=('Arial', 20), fill='white')
frame3.drop_target_register(DND_FILES)
frame3.dnd_bind('<<Drop>>', lambda e: load(e, frame3, 3))  # Corretto: frame3 e path_var=3


# Text widget PRIMA del bottone
text_widget = tk.Text(window, height=5, width=80)
text_widget.grid(row=1, column=0)

frameButton = tk.Frame(window)
frameButton.config(bg='gray')
frameButton.grid(row=2, column=0, pady=10)

def run_kontext():
    prompt_text = text_widget.get('1.0', tk.END).strip()
    
    # Controllo minimo: path1 (depth) e path2 (riferimento1) sono obbligatori
    if not path1:
        print("❌ Errore: carica l'immagine per la depth map (frame 1)!")
        return
    
    if not path2:
        print("❌ Errore: carica almeno l'immagine di riferimento 1 (frame 2)!")
        return
    
    if not prompt_text:
        print("❌ Errore: inserisci un prompt testuale!")
        return
    
    # Path3 è opzionale
    if path3:
        print("🚀 Avvio generazione con: Depth + Riferimento1 + Riferimento2")
        kontext(path1, path2, path3, prompt_text)
    else:
        print("🚀 Avvio generazione con: Depth + Riferimento1")
        kontext(path1, path2, None, prompt_text)

bntkontext = tk.Button(frameButton, text='Genera Kontext', bg='green', fg='white', 
                        font=('Arial', 14, 'bold'), command=run_kontext, padx=2, pady=2)
bntkontext.grid(row=0, column=0, padx=(0, 1))

import torch
from diffusers import ZImagePipeline

def f_texttoImage():
    global text_widget
    pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    )
    pipe.enable_model_cpu_offload()

    prompt = GoogleTranslator(source='it',target='en').translate(text_widget.get('1.0',tk.END))
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(42),
    ).images[0]
    image.save("zimage.png")
    print("✅ Immagine salvata come zimage")

bnttexttoimage = tk.Button(frameButton, text='Z_Image', bg='Orange', fg='white', font=('Arial', 14, 'bold'), command=f_texttoImage, padx=2, pady=2)
bnttexttoimage.grid(row=0, column=1, padx=(1, 0))

def f_inpainting():
    print("inpainting")
    global text_widget, path1, path2, path3, lora, scale_lora, steps, cfg, modifica
    
    # Determina il path dell'immagine
    path = None
    if path1 is not None and os.path.exists(path1):
        path = path1
    elif path2 is not None and os.path.exists(path2):
        path = path2
    elif path3 is not None and os.path.exists(path3):
        path = path3
    else:
        print("❌ Errore: Nessuna immagine valida trovata!")
        return
    
    # Ottieni il prompt e rimuovi spazi/newline in eccesso
    prompt = text_widget.get('1.0', tk.END).strip()
    
    # Costruisci il comando
    cmd = f'python inpa.py --mode 1 --path_image "{path}" --path_mask mask.jpg --prompt "{prompt}"'
    
    # Aggiungi LoRA solo se esiste
    if lora.get():  # Se è stato selezionato qualcosa nella combobox
        lora_name = lora.get()
        cmd += f' --lora "{lora_name}" --scale_lora {scale_lora.get()}'
    
    # Aggiungi parametri di generazione
    cmd += f' --steps {int(steps.get())} --cfg {cfg.get()} --strength {modifica.get()} --output inpaint'
    
    print(f"Esecuzione comando:\n{cmd}")
    os.system(cmd)

bntInpainting = tk.Button(frameButton, text='Inpainting', bg='dark orange', fg='white', font=('Arial', 14, 'bold'), command=f_inpainting, padx=2, pady=2)
bntInpainting.grid(row=0, column=2, padx=(1, 0))

def f_Fill():
    print("Fill")
    global text_widget, path1, path2, path3, lora, scale_lora, steps, cfg, modifica
    
    # Trova primo path valido
    path = next((p for p in [path1, path2, path3] if p and os.path.exists(p)), None)
    if not path:
        print("❌ Errore: Nessuna immagine valida!")
        return
    
    # Ottieni il prompt
    prompt = text_widget.get('1.0', tk.END).strip()
    
    # Costruisci comando - mode 2 per Fill
    # Se CFG è ancora al default (6.0), usa 30, altrimenti usa il valore dell'utente
    cfg_value = 30 if abs(cfg.get() - 6.0) < 0.01 else cfg.get()
    
    cmd = f'python inpa.py --mode 2 --path_image "{path}" --path_mask mask.jpg --prompt "{prompt}" --steps {int(steps.get())} --cfg {cfg_value} --strength {modifica.get()} --output fill'
    
    # Aggiungi LoRA solo se selezionata
    if lora.get():
        cmd += f' --lora "{lora.get()}" --scale_lora {scale_lora.get()}'
    
    print(f"Comando Fill: {cmd}")
    os.system(cmd)
    
bntFill = tk.Button(frameButton, text='Fill', bg='dark green', fg='white', font=('Arial', 14, 'bold'), command=f_Fill, padx=2, pady=2)
bntFill.grid(row=0, column=3, padx=(1, 0))

def f_INPA_CONTROL():
    print("Inpaint control")
    global text_widget, path1, path2, path3, lora, scale_lora, steps, cfg, modifica
    
    # Trova primo path valido
    path = next((p for p in [path1, path2, path3] if p and os.path.exists(p)), None)
    if not path:
        print("❌ Errore: Nessuna immagine valida!")
        return
    
    # Ottieni il prompt
    prompt = text_widget.get('1.0', tk.END).strip()
    
    # Costruisci comando - mode 3 per ControlNet
    # Se CFG è ancora al default (6.0), usa 7.5, altrimenti usa il valore dell'utente
    cfg_value = 7.5 if abs(cfg.get() - 6.0) < 0.01 else cfg.get()
    
    cmd = f'python inpa.py --mode 3 --path_image "{path}" --path_mask mask.jpg --prompt "{prompt}" --steps {int(steps.get())} --cfg {cfg_value} --strength {modifica.get()} --output inpa_control'
    
    # Aggiungi LoRA solo se selezionata
    if lora.get():
        cmd += f' --lora "{lora.get()}" --scale_lora {scale_lora.get()}'
    
    print(f"Comando inpainting control: {cmd}")
    os.system(cmd)

bntIpa_control = tk.Button(frameButton, text='INPA_CONTROL', bg='gold', fg='black', font=('Arial', 14, 'bold'), command=f_INPA_CONTROL, padx=2, pady=2)
bntIpa_control.grid(row=0, column=4, padx=(1, 0))

# Label per gli slider con valori dinamici
tk.Label(frameButton, text='Steps:', font=('Arial', 10)).grid(row=1, column=0, sticky='w', padx=(5, 0))
steps_value_label = tk.Label(frameButton, text='30', font=('Arial', 10, 'bold'), fg='blue')
steps_value_label.grid(row=1, column=0, sticky='e', padx=(0, 5))
steps = ttk.Scale(frameButton, from_=1, to=50, orient='horizontal', length=150,
                  command=lambda v: steps_value_label.config(text=f'{int(float(v))}'))
steps.set(30)
steps.grid(row=2, column=0, padx=(5, 5))

tk.Label(frameButton, text='CFG:', font=('Arial', 10)).grid(row=1, column=1, sticky='w', padx=(5, 0))
cfg_value_label = tk.Label(frameButton, text='6.00', font=('Arial', 10, 'bold'), fg='blue')
cfg_value_label.grid(row=1, column=1, sticky='e', padx=(0, 5))
cfg = ttk.Scale(frameButton, from_=1.0, to=30.0, orient='horizontal', length=150,
                command=lambda v: cfg_value_label.config(text=f'{float(v):.2f}'))
cfg.set(6.0)
cfg.grid(row=2, column=1, padx=(5, 5))

tk.Label(frameButton, text='Strength:', font=('Arial', 10)).grid(row=1, column=2, sticky='w', padx=(5, 0))
modifica_value_label = tk.Label(frameButton, text='0.80', font=('Arial', 10, 'bold'), fg='blue')
modifica_value_label.grid(row=1, column=2, sticky='e', padx=(0, 5))
modifica = ttk.Scale(frameButton, from_=0.0, to=1.0, orient='horizontal', length=150,
                     command=lambda v: modifica_value_label.config(text=f'{float(v):.2f}'))
modifica.set(0.80)
modifica.grid(row=2, column=2, padx=(5, 5))

# Funzione per aggiornare la lista LoRA
def update_lora_list(event=None):
    if os.path.exists('./Lora'):
        lora_files = [f for f in os.listdir('./Lora') if f.endswith('.safetensors')]
        lora['values'] = lora_files
    else:
        lora['values'] = []

tk.Label(frameButton, text='LoRA:', font=('Arial', 10)).grid(row=0, column=5, sticky='w', padx=(5, 0))
lora = ttk.Combobox(frameButton, width=20, state='readonly')
lora.grid(row=0, column=5, padx=(5, 5))
lora.bind('<Button-1>', update_lora_list)

tk.Label(frameButton, text='LoRA Weight:', font=('Arial', 10)).grid(row=1, column=5, sticky='w', padx=(5, 0))
scale_lora_value_label = tk.Label(frameButton, text='0.80', font=('Arial', 10, 'bold'), fg='blue')
scale_lora_value_label.grid(row=1, column=5, sticky='e', padx=(0, 5))
scale_lora = ttk.Scale(frameButton, from_=0.10, to=1.0, orient='horizontal', length=150,
                       command=lambda v: scale_lora_value_label.config(text=f'{float(v):.2f}'))
scale_lora.set(0.80)
scale_lora.grid(row=2, column=5, padx=(5, 5))

import torch
import numpy as np
import os
import tkinter as tk
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import urllib.request

# Funzione per scaricare il file BPE
def download_bpe_vocab():
    assets_dir = r"G:\Edit depth\vdepth\lib\site-packages\assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    bpe_path = os.path.join(assets_dir, "bpe_simple_vocab_16e6.txt.gz")
    
    if os.path.exists(bpe_path):
        print("✅ File BPE già presente")
        return
    
    print("📥 Scaricamento file BPE vocabulary...")
    url = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
    
    try:
        urllib.request.urlretrieve(url, bpe_path)
        print(f"✅ BPE salvato in: {bpe_path}")
    except Exception as e:
        print(f"❌ Errore download BPE: {e}")

# Scarica il file BPE all'avvio (se necessario)
download_bpe_vocab()
import cv2

def f_mask():
    global path1, path2, path3, text_widget,confidenza,pixels
    t = text_widget.get('1.0', tk.END)
    t = GoogleTranslator(source='it', target='en').translate(t)

    # Trova il percorso valido
    path = None
    for p in [path1, path2, path3]:
        if os.path.exists(p):
            path = p
            break
    
    if path is None:
        print("❌ Nessun percorso valido trovato.")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Usando device: {device}")
    
    try:
        print("📦 Caricamento modello SAM3...")
        
        # Carica il modello (scarica automaticamente da Hugging Face se necessario)
        model = build_sam3_image_model()
        processor = Sam3Processor(model)

        # Carica l'immagine
        print(f"🖼️ Caricamento immagine: {path}")
        image_pil = Image.open(path).convert("RGB")
        width, height = image_pil.size
        
        inference_state = processor.set_image(image_pil)

        clothing_list = (
            "underwear, bra, panties, thong, lingerie, bodysuit, corset, bralette, "
            "bikini, swimsuit, monokini, tankini, sarong, dress, maxi dress, "
            "sundress, evening gown, cocktail dress, wrap dress, skirt, mini skirt, "
            "pencil skirt, pleated skirt, pants, leggings, skinny jeans, trousers, "
            "shorts, hot pants, crop top, blouse, shirt, stockings, garter, robe, pajamas, "
            "white top, black pants, woman's clothing"
        )
        
        prompt_text = clothing_list if t.strip() == '' else t
        print(f"🔍 Ricerca prompt: {prompt_text[:50]}...")
        
        # Rimosso multimask_output - non è supportato
        output = processor.set_text_prompt(
            state=inference_state, 
            prompt=prompt_text
        )

        masks = output["masks"]
        scores = output["scores"]
        
        print(f"📊 Ricevute {len(masks)} maschere con scores: {scores}")
        print(f"📊 Shape maschere: {masks.shape}")

        # Filtra per confidenza
        c = float(confidenza.get())
        high_conf_masks = masks[scores > c]

        if len(high_conf_masks) > 0:
            print(f"✅ Trovate {len(high_conf_masks)} maschere con confidenza > {c}")
            
            # Combina le maschere
            combined_mask = torch.any(high_conf_masks, dim=0)
            
            # Rimuovi dimensioni extra e converti in numpy
            combined_mask = combined_mask.squeeze().cpu().numpy()
            
            print(f"📊 Shape maschera finale: {combined_mask.shape}")
            
            # Assicurati che sia 2D
            if combined_mask.ndim > 2:
                # Se ha ancora più di 2 dimensioni, prendi solo il primo canale
                combined_mask = combined_mask[0] if combined_mask.shape[0] == 1 else combined_mask
            
            # Converti in 0-255
            mask_image = (combined_mask * 255).astype(np.uint8)
            
            # ESPANSIONE DEI BORDI BIANCHI DI 22 PIXEL
            # Crea un kernel circolare per la dilatazione
            kernel_size =int(pixels.get())
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size*2+1, kernel_size*2+1))
            
            # Applica la dilatazione morfologica
            mask_image = cv2.dilate(mask_image, kernel, iterations=1)
            print(f"✅ Maschera espansa di {kernel_size} pixel")
            
            # Crea l'immagine PIL
            final_mask_pil = Image.fromarray(mask_image, mode='L')  # 'L' per grayscale
            
            # Ridimensiona alle dimensioni originali
            final_mask_pil = final_mask_pil.resize((width, height), resample=Image.NEAREST)

            save_path = "mask.jpg"
            final_mask_pil.save(save_path)
            print(f"💾 Maschera salvata: {save_path}")
        else:
            print(f"❌ Nessun indumento rilevato con confidenza > {c}")
            print("💡 Prova ad abbassare la soglia di confidenza o usa un prompt più specifico")
    
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

detectoutfit = tk.Button(frameButton, text='mask Outfit', bg='light blue', command=f_mask)
detectoutfit.grid(row=0, column=6, sticky='e', padx=(0, 5))

import os
import tkinter as tk
import webbrowser
from tkinter import messagebox

def F_mask_m():
    print("Crea Maschera manuale")
    
    dirs = ['C:\\Program Files', 'C:\\Program Files (x86)']
    path = None
    
    for base_dir in dirs:
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                if item == 'Adobe':
                    adobe_path = os.path.join(base_dir, item)
                    # Cerca nelle sottocartelle di Adobe
                    for folder in os.listdir(adobe_path):
                        if "photoshop" in folder.lower():
                            photoshop_folder = os.path.join(adobe_path, folder)
                            # Cerca l'eseguibile Photoshop.exe
                            for file in os.listdir(photoshop_folder):
                                if file.lower() == "photoshop.exe":
                                    path = os.path.join(photoshop_folder, file)
                                    break
                        if path:
                            break
                if path:
                    break
        if path:
            break
    
    if path:
        print(f"Photoshop trovato: {path}")
        os.startfile(path)
        
        if not os.path.exists("dont.bat"):
            # Finestra tutorial
            tutorial_window = tk.Toplevel()
            tutorial_window.title("Tutorial Maschera Manuale")
            tutorial_window.geometry("500x350")
            
            # Variabile per checkbox
            show_again = tk.BooleanVar(value=False)
            
            messaggio = """
            TUTORIAL CREA MASCHERA MANUALE CON PHOTOSHOP
            
            1. Crea Nuovo livello (+)
            2. Seleziona Indumenti (selezione oggetto, lazo, altro)
            3. Riempi con tinta bianca (barattolo, seleziona colore bianco)
            4. Crea nuovo livello (+)
            5. Riempi con tinta nera (barattolo, seleziona colore nero)
            6. Sposta livello tinta nera sotto il livello tinta bianca
            7. Save As: 'mask.jpg'
            """
            
            # Label con il messaggio
            label = tk.Label(tutorial_window, text=messaggio, justify=tk.LEFT, padx=20, pady=20)
            label.pack()
            
            # Checkbox
            ck = tk.Checkbutton(tutorial_window, text='Non mostrare questo messaggio', 
                               variable=show_again)
            ck.pack(pady=10)
            
            # Funzione per chiudere e salvare preferenza
            def close_tutorial():
                if show_again.get():
                    with open("dont.bat", 'w') as f:
                        f.write("# Non mostrare tutorial")
                tutorial_window.destroy()
            
            # Bottone OK
            btn_ok = tk.Button(tutorial_window, text='OK', 
                              command=close_tutorial, width=10)
            btn_ok.pack(pady=10)
        
    else:
        # Se Photoshop non è trovato, apri il sito Adobe
        print("Photoshop non trovato")
        risposta = messagebox.askyesno(
            "Photoshop non trovato",
            "Photoshop non è installato. Vuoi aprire il sito Adobe per scaricarlo?"
        )
        if risposta:
            webbrowser.open("https://www.adobe.com/it/products/photoshop.html")


mask_manual=tk.Button(frameButton,text='Mask_manual',bg='light green',command=F_mask_m)
mask_manual.grid(row=0,column=7)

def f_open_site(event=None):
    print("siti utili")
    selected = siti_utili.get()  # Ottieni il valore selezionato
    
    if selected == 'NANOBANANNAPRO_higgsfield':
        webbrowser.open("https://higgsfield.ai/image/nano_banana_2")
    elif selected == 'NANOBANANNAPRO_A2E':
        webbrowser.open("https://video.a2e.ai/image-generator/nano-banana")
    elif selected == 'ImageToVideo_A2E':
        webbrowser.open("https://video.a2e.ai/image-to-video")
    elif selected == 'ImageTovideo_VN':
        webbrowser.open("https://virtuallynude.ai/dashboard")
    elif selected == 'grok_Image_video':
        webbrowser.open("https://grok.com/imagine")
    elif selected == 'digen_image_to video':
        webbrowser.open("https://digen.ai/create")
    elif selected == 'meta.ia':
        webbrowser.open("https://www.meta.ai/media?locale=it_IT")
    elif selected == 'None':
        print("Nessun sito selezionato")


# Combobox con bind corretto
siti_utili = ttk.Combobox(frameButton, values=['None', 'NANOBANANNAPRO_higgsfield', 
                                                'NANOBANANNAPRO_A2E', 'ImageToVideo_A2E', 
                                                'ImageTovideo_VN','grok_Image_video','digen_image_to video','meta.ia'])
siti_utili.current(0)  # Imposta 'None' come valore di default
siti_utili.bind('<<ComboboxSelected>>', f_open_site)  # Usa bind invece di command
siti_utili.grid(row=0, column=8)

def estraiframe():
    print("estrai frame")
    if not os.path.exists("estrai_opening.bat"):
        with open("estrai_opening.bat", "w") as f:
            pass  # Crea il file vuoto
        os.system("python estrai.py")

estrai_frame = tk.Button(frameButton, text='Estrai Frame', bg='plum', command=estraiframe)
estrai_frame.grid(row=0, column=9)

def close():
    if os.path.exists("estrai_opening.bat"):
        os.remove("estrai_opening.bat")
    window.destroy()  # Chiude la finestra

window.protocol("WM_DELETE_WINDOW", close)
         



# Crea le Scale prima di usarle
confidenza = ttk.Scale(frameButton, from_=0.1, to=1.0, orient='horizontal')
confidenza.set(0.1)  # Valore iniziale
confidenza.grid(row=1, column=6, sticky='ew', padx=(0, 5))

pixels = ttk.Scale(frameButton, from_=1, to=30, orient='horizontal')
pixels.set(21)  # Valore iniziale
pixels.grid(row=1, column=7, sticky='ew', padx=(0, 5))

# Label per mostrare i valori
label_confidenza = tk.Label(frameButton, text=f"Confidenza: {confidenza.get():.1f}")
label_confidenza.grid(row=2, column=6, sticky='w', padx=(0, 5))

label_pixels = tk.Label(frameButton, text=f"Pixels: {int(pixels.get())}")
label_pixels.grid(row=2, column=7, sticky='w', padx=(0, 5))

# Funzione per aggiornare le label quando gli slider cambiano
def update_labels(val):
    label_confidenza.config(text=f"Confidenza: {confidenza.get():.1f}")
    label_pixels.config(text=f"Pixels: {int(pixels.get())}")

confidenza.config(command=update_labels)
pixels.config(command=update_labels)

window.mainloop()