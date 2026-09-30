import argparse
from pathlib import Path
from PIL import Image
import numpy as np


def main():
	parser = argparse.ArgumentParser(description="Convert an image to RGB565 data.")
	parser.add_argument("input", nargs="?", default="example.png", help="input image (default: example.png)")
	parser.add_argument("output", nargs="?", help="output file (default: input filename with .rgb565 extension)")
	args = parser.parse_args()

	input_path = Path(args.input)
	output_path = Path(args.output) if args.output else input_path.with_suffix(".rgb565")

	img = Image.open(input_path).convert("RGB")
	a = np.array(img, dtype=np.uint8)
	r = (a[..., 0] >> 3).astype(np.uint16)
	g = (a[..., 1] >> 2).astype(np.uint16)
	b = (a[..., 2] >> 3).astype(np.uint16)
	rgb565 = (r << 11) | (g << 5) | b

	# Match the byte order expected by the device.
	rgb565.astype("<u2").tofile(output_path)


if __name__ == "__main__":
	main()
