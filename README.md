un modello che tramite un lora depthmap collegato con kontext riesce a calcolare posa e profondita, di un oggetto o character, contiene anche la funzione fill, inpainting, di sd e flux, e differenti collegamenti a servizi freen Nano bananna e Image to video
git clone: https://github.com/asprho-arkimete/Multi-edit-depthmap-fill-video-generation.git
cd Multi-edit-depthmap-fill-video-generation
python 3.10 -m venv vmulti
cd vmulti\\Scripts\\activate
cd.. cd..
scarica il file lora da : https://huggingface.co/thedeoxen/FLUX.1-Kontext-dev-reference-depth-fusion-LORA/tree/main
scarica e istalla Daz3d per i modelli 3d per ricavare le depth map: https://www.daz3d.com/
altri modelli trovi il files txt nelle directory Model e Lora

istalla le dipendenze: 
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
