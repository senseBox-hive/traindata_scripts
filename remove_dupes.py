import argparse
import os, re, shutil
import glob
from collections import defaultdict

WORK_DIR = "."
CROP_SIZE = 32
THRESHOLD = 0.25
TO_DELETE_DIR = "to_delete"

def extract_metadata(filename):
    # information stored in the title
    exp = "(.*)__x([0-9]+)_y([0-9]+)_a([0-9]+)\\.(?:png|jpg)"
    match = re.search(exp, filename)
    if match is None:
        raise ValueError(f"Filename does not contain crop metadata: {filename}")

    frame_id = match.group(1)
    x = int(match.group(2))
    y = int(match.group(3))
    area = int(match.group(4)) #probably not a useful value. The area of saturation that was detected


    return {
        "filename": filename,
        "frame_id": frame_id,
        "x": x,
        "y": y,
        "area": area,
        "width": CROP_SIZE,
        "height": CROP_SIZE
    }

def check_overlap(img_a, img_b):
    #we have centroid, height and width
    cent_a = (img_a["x"], img_a["y"])
    cent_b = (img_b["x"], img_b["y"])

    # axis overlaps
    dist_x = abs(cent_a[0] - cent_b[0])
    dist_y = abs(cent_a[1] - cent_b[1])

    overlap_x = max(0,(img_a["width"]/2 + img_b["width"]/2) - dist_x)
    overlap_y = max(0,(img_a["height"]/2 + img_b["height"]/2) - dist_y)

    overlap_area = overlap_x * overlap_y

    # this returns the overlap ratio of A on B. if both frames are the same size it wont matter
    overlap_ratio = overlap_area / (img_a["height"] * img_a["width"])
    return overlap_ratio

# look at the existing frames and return lists of crops that are in the same frame
def group_frames(paths):
    if not paths:
        print(f"No images found in '{WORK_DIR}/'")
        return []

    files = []
    for p in paths:
        files.append(extract_metadata(p))
    print("extracted metadata")

    groups = []
    groupnames = set(map(lambda x: x["frame_id"], files))
    groupnames = [*groupnames]
    for i in enumerate(groupnames):
        groups.append([])
    for f in files:
        i = groupnames.index(f["frame_id"])
        groups[i].append(f)
    print(f"grouped {len(paths)} frames into {len(groups)} framegroups")

    # prune solitary samples
    groups = list(filter(lambda x : len(x) > 1, groups))
    print(f"reduced to {len(groups)} framegroups")
    return groups

def remove_overlapping_squares(squares, overlap_fraction, threshold=0.25,
                                exact_max_component=22):
    """
    VIBECODED: shame on me

    Determine which squares to keep so no two overlap by more than `threshold`
    of their own area, while removing as few squares as possible.

    Parameters
    ----------
    squares : list
        Your list of square objects (any type).
    overlap_fraction : callable
        Function overlap_fraction(a, b) -> float in [0, 1], the fraction of
        overlap relative to a square's area. Assumes symmetric (same-sized squares).
    threshold : float
        Max allowed overlap fraction (default 0.25 = 25%).
    exact_max_component : int
        Max component size to solve exactly (branch & bound). Larger uses greedy.

    Returns
    -------
    keep_indices : set[int]   Indices of squares to KEEP
    remove_indices: set[int]  Indices of squares to REMOVE
    """
    n = len(squares)

    # ---------- Step 2: Build conflict graph (pairwise) ----------
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            print(squares[i])
            print(squares[j])
            print(overlap_fraction(squares[i], squares[j]))
            print("##########")
            if overlap_fraction(squares[i], squares[j]) > threshold:
                adj[i].add(j)
                adj[j].add(i)

    # ---------- Step 3: Connected components ----------
    components = _connected_components(n, adj)

    # ---------- Step 4: Solve Maximum Independent Set per component ----------
    keep_indices = set()
    for comp in components:
        if len(comp) == 1:
            keep_indices.add(next(iter(comp)))
        elif len(comp) <= exact_max_component:
            keep_indices |= _mis_exact(comp, adj)
        else:
            keep_indices |= _mis_greedy(comp, adj)

    remove_indices = set(range(n)) - keep_indices
    return keep_indices, remove_indices


def _connected_components(n, adj):
    """Return list of components, each a set of node indices."""
    seen = [False] * n
    components = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = set()
        while stack:
            u = stack.pop()
            comp.add(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        components.append(comp)
    return components


def _mis_greedy(nodes, adj):
    """Greedy min-degree Maximum Independent Set (fast, approximate)."""
    nodes = set(nodes)
    keep = set()
    while nodes:
        # pick the node with fewest remaining neighbors
        u = min(nodes, key=lambda x: len(adj[x] & nodes))
        keep.add(u)
        nodes.discard(u)
        nodes -= adj[u]           # remove its neighbors
    return keep


def _mis_exact(nodes, adj):
    """
    Exact Maximum Independent Set via branch & bound.
    Suitable for small components (bounded by exact_max_component).
    """
    nodes = set(nodes)
    best = set()

    def neighbors_in(u, remaining):
        return adj[u] & remaining

    def branch(remaining, current):
        nonlocal best
        # Bound: even taking everything left can't beat best
        if len(current) + len(remaining) <= len(best):
            return
        if not remaining:
            if len(current) > len(best):
                best = set(current)
            return

        # Pick pivot: highest-degree node to branch on
        v = max(remaining, key=lambda x: len(neighbors_in(x, remaining)))

        # Branch 1: EXCLUDE v
        branch(remaining - {v}, current)

        # Branch 2: INCLUDE v (removes v and its neighbors)
        branch(remaining - {v} - adj[v], current | {v})

    branch(nodes, set())
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Move overlapping crop images into a to_delete subdirectory."
    )
    parser.add_argument("--work-dir", default=WORK_DIR, help=f"directory containing crop images (default: {WORK_DIR})")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help=f"maximum allowed overlap fraction (default: {THRESHOLD})")
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    work_dir = os.path.abspath(args.work_dir)
    to_delete_dir = os.path.join(work_dir, TO_DELETE_DIR)

    paths = sorted(
        glob.glob(os.path.join(work_dir, "*.jpg")) +
        glob.glob(os.path.join(work_dir, "*.png"))
    )
    if not paths:
        print(f"No images found in '{work_dir}/'")
        return

    framegroups = group_frames(paths)
    paths_to_delete = []

    print(f"moving overlaps of >{args.threshold} to '{to_delete_dir}/'")
    for group in framegroups:
        keep_indices, remove_indices = remove_overlapping_squares(
            group,
            check_overlap,
            threshold=args.threshold
            )
        paths_to_delete.extend(group[i]["filename"] for i in remove_indices)
        print(f"reduced {group[0]['frame_id']} from {len(group)} to {len(keep_indices)}")

    os.makedirs(to_delete_dir, exist_ok=True)
    for path in paths_to_delete:
        shutil.move(path, os.path.join(to_delete_dir, os.path.basename(path)))

    print(f"\nDone. Kept {len(paths) - len(paths_to_delete)} frames in '{work_dir}/'; moved {len(paths_to_delete)} to '{to_delete_dir}/'.")

if __name__ == "__main__":
    main()

