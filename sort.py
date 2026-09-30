import argparse
import os, shutil, tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

CLASS_BINDINGS = (
    "1: bee_motion",
    "2: bee_resting",
    "3: background",
    "4: discard",
)


def main():
    parser = argparse.ArgumentParser(description="Sort image files into class directories.")
    parser.add_argument("--input-dir", default="unsorted", help="directory containing images (default: unsorted)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"input directory does not exist: {input_dir}")

    files = sorted(path for path in input_dir.iterdir() if path.is_file())
    if not files:
        print(f"No files found in '{input_dir}'")
        return

    idx = 0
    root = tk.Tk()
    lbl = tk.Label(root)
    lbl.pack()

    def show():
        nonlocal idx
        img = Image.open(files[idx]).resize((640, 640))
        tkimg = ImageTk.PhotoImage(img)
        lbl.config(image=tkimg)
        lbl.image = tkimg

        print("\033[2J\033[H", end="")
        print(f"{files[idx].name}\n")
        print("Class bindings:")
        for binding in CLASS_BINDINGS:
            print(f"  {binding}")
        print(flush=True)

    def classify(cls):
        nonlocal idx
        shutil.move(str(files[idx]), str(Path(cls) / files[idx].name))
        idx += 1
        if idx < len(files):
            show()
        else:
            root.quit()

    root.bind("1", lambda e: classify("bee_motion"))
    root.bind("2", lambda e: classify("bee_slow"))
    root.bind("3", lambda e: classify("background"))
    root.bind("4", lambda e: classify("discard"))
    show()
    root.mainloop()


if __name__ == "__main__":
    main()
