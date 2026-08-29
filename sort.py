import os, shutil, tkinter as tk
from PIL import Image, ImageTk

files = [f for f in os.listdir("unsorted")]
idx = 0
root = tk.Tk()
lbl = tk.Label(root); lbl.pack()

def show():
    img = Image.open(f"unsorted/{files[idx]}").resize((640,640))  # 20x upscale
    tkimg = ImageTk.PhotoImage(img)
    lbl.config(image=tkimg); lbl.image = tkimg

def classify(cls):
    global idx
    shutil.move(f"unsorted/{files[idx]}", f"{cls}/{files[idx]}")
    idx += 1
    if idx < len(files): show()
    else: root.quit()

root.bind("1", lambda e: classify("bee_motion"))
root.bind("2", lambda e: classify("bee_resting"))
root.bind("3", lambda e: classify("background"))
root.bind("4", lambda e: classify("others"))
show(); root.mainloop()
