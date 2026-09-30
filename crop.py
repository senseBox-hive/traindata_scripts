import argparse
import cv2, numpy as np, os
import glob

INPUT_DIR   = "uncropped"
OUTPUT_DIR  = "unsorted"

MAX_CROPS   = 1000        # frames yielding more than this are discarded

CROP_SIZE   = 32          # output patch is CROP_SIZE x CROP_SIZE (square)
SAT_THRESH  = 40          # saturation (colour-purity) threshold, 0-255.
MAX_SAT     = 120
                          # None => auto-estimate from image noise floor.
MIN_AREA    = 16           # ignore blobs smaller than this (px)
MAX_AREA    = 800          # ignore blobs larger than this (likely vegetation/lighting)
PAD_EDGE    = True        # if True, pad crops that fall off the image edge
                          # if False, skip blobs too close to the border
BLUR_KSIZE  = 3           # optional pre-blur of the saturation map (0 = off, else odd, e.g. 3)

def compute_saturation(bgr):
    """Colour-purity map: max(channel) - min(channel).
    Grey (static background) -> ~0.  Pure single-channel dot -> high."""
    mx = np.max(bgr, axis=2).astype(np.int16)
    mn = np.min(bgr, axis=2).astype(np.int16)
    sat = (mx - mn)
    return np.clip(sat, 0, 255).astype(np.uint8)

def estimate_threshold(sat):
    """Auto threshold from the saturation noise floor.
    Assumes most of the frame is static background (low saturation)."""
    mean = float(np.mean(sat))
    std  = float(np.std(sat))
    thr  = int(mean + 5.0 * std)          # 5-sigma above background
    return max(thr, 10)                    # never go below a sane floor

def crop_patch(img, cx, cy, size, pad_edge):
    """Return a size x size crop centred on (cx, cy), or None."""
    half = size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + size, y0 + size
    h, w = img.shape[:2]

    if x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h:
        return img[y0:y1, x0:x1].copy()

    if not pad_edge:
        return None

    # blob near border: pad with reflect so the crop stays size x size
    patch = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, w), min(y1, h)
    dx0, dy0 = sx0 - x0, sy0 - y0
    patch[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return patch


def process_frame(path, output_dir, sat_threshold, min_area, max_area):
    img = cv2.imread(path, cv2.IMREAD_COLOR)   # BGR
    if img is None:
        print(f"  !! could not read {path}")
        return 0, 0, 0, False

    stem = os.path.splitext(os.path.basename(path))[0]

    sat = compute_saturation(img)
    if BLUR_KSIZE >= 3:
        sat = cv2.GaussianBlur(sat, (BLUR_KSIZE, BLUR_KSIZE), 0)

    thr = sat_threshold if sat_threshold is not None else estimate_threshold(sat)
    _, mask = cv2.threshold(sat, thr, MAX_SAT, cv2.THRESH_BINARY)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

    pending = []                                # (out_name, patch, cx, cy)
    print("total blobs: ", n)
    for i in range(1, n):                       # 0 is background
        area   = stats[i, cv2.CC_STAT_AREA]
        width  = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        if area < min_area or area > max_area:
            continue
        if width > CROP_SIZE or height > CROP_SIZE:
            continue

        cx, cy = centroids[i]
        cx, cy = int(round(cx)), int(round(cy))

        patch = crop_patch(img, cx, cy, CROP_SIZE, PAD_EDGE)
        if patch is None:
            continue

        out_name = f"{stem}__x{cx}_y{cy}_a{area}.png"
        pending.append((out_name, patch, cx, cy))

        # early exit: no point cropping 500 blobs we're about to throw away
        if len(pending) > MAX_CROPS:
            return n, 0, len(pending), True

    preview = None

    for out_name, patch, cx, cy in pending:
        cv2.imwrite(os.path.join(output_dir, out_name), patch)
        if preview is not None:
            cv2.rectangle(preview,
                          (cx - CROP_SIZE // 2, cy - CROP_SIZE // 2),
                          (cx + CROP_SIZE // 2, cy + CROP_SIZE // 2),
                          (0, 255, 0), 1)

    if preview is not None:
        os.makedirs("preview", exist_ok=True)
        cv2.imwrite(os.path.join("preview", f"{stem}_preview.png"), preview)
    return n, len(pending), len(pending), False


def main():
    max_n = 0
    parser = argparse.ArgumentParser(description="create crops of candidate blobs.")
    parser.add_argument("--input-dir", default=INPUT_DIR, help=f"source dir (default: {INPUT_DIR})")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help=f"output dir (default: {OUTPUT_DIR})")
    parser.add_argument("--sat-threshold", type=int, default=SAT_THRESH, help=f"saturation threshold (0-255, default: {SAT_THRESH})")
    parser.add_argument("--min-area", type=int, default=MIN_AREA, help=f"minimum blob area (default: {MIN_AREA})")
    parser.add_argument("--max-area", type=int, default=MAX_AREA, help=f"maximum blob area (default: {MAX_AREA})")
    args = parser.parse_args()

    if not 0 <= args.sat_threshold <= 255:
        parser.error("--sat-threshold must be between 0 and 255")
    if args.min_area < 0 or args.max_area < 0:
        parser.error("--min-area and --max-area must be non-negative")
    if args.min_area > args.max_area:
        parser.error("--min-area cannot be greater than --max-area")

    os.makedirs(args.output_dir, exist_ok=True)

    paths = sorted(
        glob.glob(os.path.join(args.input_dir, "*.png")) +
        glob.glob(os.path.join(args.input_dir, "*.jpg"))
    )
    if not paths:
        print(f"No images found in '{args.input_dir}/'")
        return

    total = 0
    rejected_frames = 0
    rejected_crops = 0

    for p in paths:
        nmax, written, found, rejected = process_frame(
            p,
            args.output_dir,
            args.sat_threshold,
            args.min_area,
            args.max_area,
        )
        if nmax > max_n:
            max_n = nmax
        total += written
        if rejected:
            rejected_frames += 1
            rejected_crops += found
            print(f"{os.path.basename(p):40s} -> SKIPPED ({found}+ blobs > MAX_CROPS={MAX_CROPS})")
        else:
            print(f"{os.path.basename(p):40s} -> {written} crops")

    print("max blobs: ", max_n)

    print(f"\nDone. {total} crops written to '{args.output_dir}/'")
    if rejected_frames:
        print(f"Discarded {rejected_frames} frame(s) "
              f"({rejected_crops}+ blobs) for exceeding MAX_CROPS={MAX_CROPS}")


if __name__ == "__main__":
    main()
