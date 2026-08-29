import cv2, numpy as np, os
import glob
import cv2

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
INPUT_DIR   = "uncropped"
OUTPUT_DIR  = "unsorted"

CROP_SIZE   = 32          # output patch is CROP_SIZE x CROP_SIZE (square)
SAT_THRESH  = 40          # saturation (colour-purity) threshold, 0-255.
                          # None => auto-estimate from image noise floor.
MIN_AREA    = 16           # ignore blobs smaller than this (px)
MAX_AREA    = 800          # ignore blobs larger than this (likely vegetation/lighting)
PAD_EDGE    = True        # if True, pad crops that fall off the image edge
                          # if False, skip blobs too close to the border
BLUR_KSIZE  = 0           # optional pre-blur of the saturation map (0 = off, else odd, e.g. 3)
SAVE_MASK_PREVIEW = False # dump a debug overlay per frame to preview/ for tuning
# ----------------------------------------------------------------------


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


def process_frame(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)   # BGR
    if img is None:
        print(f"  !! could not read {path}")
        return 0

    stem = os.path.splitext(os.path.basename(path))[0]

    sat = compute_saturation(img)
    if BLUR_KSIZE >= 3:
        sat = cv2.GaussianBlur(sat, (BLUR_KSIZE, BLUR_KSIZE), 0)

    thr = SAT_THRESH if SAT_THRESH is not None else estimate_threshold(sat)
    _, mask = cv2.threshold(sat, thr, 255, cv2.THRESH_BINARY)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    saved = 0
    preview = img.copy() if SAVE_MASK_PREVIEW else None

    for i in range(1, n):                       # 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            continue

        cx, cy = centroids[i]
        cx, cy = int(round(cx)), int(round(cy))

        patch = crop_patch(img, cx, cy, CROP_SIZE, PAD_EDGE)
        if patch is None:
            continue

        out_name = f"{stem}__x{cx}_y{cy}_a{area}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), patch)
        saved += 1

        if preview is not None:
            cv2.rectangle(preview,
                          (cx - CROP_SIZE // 2, cy - CROP_SIZE // 2),
                          (cx + CROP_SIZE // 2, cy + CROP_SIZE // 2),
                          (0, 255, 0), 1)

    if preview is not None:
        os.makedirs("preview", exist_ok=True)
        cv2.imwrite(os.path.join("preview", f"{stem}_preview.png"), preview)

    return saved


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    paths = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.png")) +
        glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    )
    if not paths:
        print(f"No images found in '{INPUT_DIR}/'")
        return

    total = 0
    for p in paths:
        c = process_frame(p)
        total += c
        print(f"{os.path.basename(p):40s} -> {c} crops")

    print(f"\nDone. {total} crops written to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
