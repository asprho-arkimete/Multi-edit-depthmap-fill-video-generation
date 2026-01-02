import tkinter as tk
from tkinter import ttk, filedialog
from moviepy import VideoFileClip
from PIL import Image, ImageTk
import os

window = tk.Tk()
window.config(bg='gray')
window.geometry("1280x720")
window.resizable(False, False)

# Canvas più piccolo per lasciare spazio ai bottoni
frame = tk.Canvas(window, width=1024, height=600, bg='red')
frame.grid(row=0, column=0, columnspan=5, padx=10, pady=10)

clip = None
path = None

def F_load():
    global frame, path, scorre, clip
    path = filedialog.askopenfilename()
    if not path:  # Se l'utente annulla
        return
        
    clip = VideoFileClip(path)
    fb = clip.get_frame(0)  # Ottiene il primo frame come array numpy
    fb = Image.fromarray(fb.astype('uint8'))  # Converte in immagine PIL
    
    # Configura lo scorre con il numero totale di frame
    total_frames = int(clip.fps * clip.duration)
    scorre.config(to=total_frames - 1)
    
    wr, hr = 1024, 600
    w, h = fb.size
    if w >= h:
        hr = (wr * h) // w
    else:
        wr = (hr * w) // h
    
    fb = fb.resize((wr, hr), Image.BICUBIC)
    
    # Converte in PhotoImage per Tkinter
    photo = ImageTk.PhotoImage(fb)
    
    # Salva il riferimento per evitare garbage collection
    frame.image = photo
    
    # Pulisce il canvas e crea la nuova immagine
    frame.delete("all")
    frame.create_image(512, 300, image=photo)
    
    print(f"video Load: {path}")

def estraiframe(frame_index):
    global clip, path, frame
    if clip is None:
        print("Nessun video caricato!")
        return
        
    # Converti l'indice del frame in tempo
    time = frame_index / clip.fps
    fb = clip.get_frame(time)  # Ottiene il frame come array numpy
    
    os.makedirs("frame_estratti", exist_ok=True)
    fb_img = Image.fromarray(fb.astype('uint8'))  # Converte in immagine PIL
    
    base_name = os.path.basename(path).split('.')[0]
    save_path = f"frame_estratti/{base_name}_{frame_index}.png"
    
    k = 1
    while os.path.exists(save_path):
        save_path = f"frame_estratti/{base_name}_{frame_index}_{k}.png"
        k += 1
    
    fb_img.save(save_path)
    print(f"Frame salvato: {save_path}")
    
    wr, hr = 1024, 600
    w, h = fb_img.size
    if w >= h:
        hr = (wr * h) // w
    else:
        wr = (hr * w) // h
    
    fb_img = fb_img.resize((wr, hr), Image.BICUBIC)
    
    # Converte in PhotoImage per Tkinter
    photo = ImageTk.PhotoImage(fb_img)
    
    # Salva il riferimento per evitare garbage collection
    frame.image = photo
    
    # Pulisce il canvas e crea la nuova immagine
    frame.delete("all")
    frame.create_image(512, 300, image=photo)

def scorriframes(value):
    global clip, path, frame
    if clip is None:
        return
    
    frame_index = int(float(value))
    
    # Converti l'indice del frame in tempo
    time = frame_index / clip.fps
    fb = clip.get_frame(time)  # Ottiene il frame come array numpy
    
    fb_img = Image.fromarray(fb.astype('uint8'))  # Converte in immagine PIL
    
    wr, hr = 1024, 600
    w, h = fb_img.size
    if w >= h:
        hr = (wr * h) // w
    else:
        wr = (hr * w) // h
    
    fb_img = fb_img.resize((wr, hr), Image.BICUBIC)
    
    # Converte in PhotoImage per Tkinter
    photo = ImageTk.PhotoImage(fb_img)
    
    # Salva il riferimento per evitare garbage collection
    frame.image = photo
    
    # Pulisce il canvas e crea la nuova immagine
    frame.delete("all")
    frame.create_image(512, 300, image=photo)

# Frame per i bottoni
button_frame = tk.Frame(window, bg='gray')
button_frame.grid(row=1, column=0, columnspan=5, pady=10)

load = tk.Button(button_frame, text='Load Video', command=F_load, width=12)
load.grid(row=0, column=0, padx=5)

frame_begin = tk.Button(button_frame, text='Frame Iniziale', bg='lightgreen', width=12, command=lambda: estraiframe(0))
frame_begin.grid(row=0, column=1, padx=5)

scorre = ttk.Scale(button_frame, from_=0, to=1000, length=200, command=scorriframes)
scorre.grid(row=0, column=3, padx=5)

frame_current = tk.Button(button_frame, text='Frame Current', bg='orange', width=12, command=lambda: estraiframe(int(scorre.get())))
frame_current.grid(row=0, column=2, padx=5)

frame_last = tk.Button(button_frame, text='Frame Finale', bg='lightblue', width=12, command=lambda: estraiframe(int(scorre.get())))
frame_last.grid(row=0, column=4, padx=5)

def close():
    if os.path.exists("estrai_opening.bat"):
        os.remove("estrai_opening.bat")
    window.destroy()  # Chiude la finestra

window.protocol("WM_DELETE_WINDOW", close)

window.mainloop()