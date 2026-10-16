"""The object the effect comes from, as a matte: it stays in front of the effect below the emission point.

The tapped emission point sits on the object (the top of a pumpkin, a mug's rim), so it seeds a GrabCut
segmentation of the plate's first frame: definitely object just below the point, probably object further
down, definitely background at the frame's border. The connected region holding the seed is kept, holes
filled, edges feathered like a focused photographed edge. A matte that comes out implausible (tiny, most of
the frame, or not under the point) is refused, and the compositor falls back to its soft holdout.

A stronger segmenter (the CueSync Object Matte node runs SAM 2.1 inside ComfyUI) can supply the matte
instead; everything downstream only needs a float mask at plate size.
"""

from __future__ import annotations

import numpy as np

WORK_WIDTH = 480  # segment at this width: fast, and the feathered edge hides the upscale


def grabcut_matte(frame: np.ndarray, point: tuple[float, float], iterations: int = 5) -> np.ndarray | None:
    """Float matte (plate size, 0..1) of the object under ``point`` (normalised x, y), or None."""
    import cv2  # imported here: the ComfyUI nodes import cuesync without needing OpenCV

    h, w = frame.shape[:2]
    s = min(1.0, WORK_WIDTH / w)
    sw, sh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    img = cv2.resize(np.ascontiguousarray(frame[..., ::-1]), (sw, sh), interpolation=cv2.INTER_AREA)  # BGR
    ex, ey = point[0] * sw, point[1] * sh
    short = min(sw, sh)

    mask = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    below = np.hypot((xx - ex) / (0.16 * short), (yy - (ey + 0.14 * short)) / (0.16 * short)) <= 1.0
    mask[below] = cv2.GC_PR_FGD
    seed = np.hypot(xx - ex, yy - (ey + 0.05 * short)) <= max(2.0, 0.025 * short)
    mask[seed] = cv2.GC_FGD
    b = max(2, int(0.02 * short))
    mask[:b], mask[-b:], mask[:, :b], mask[:, -b:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD
    # Nothing to separate (a flat frame, or an object the colour of its surroundings): GrabCut's colour
    # models degenerate there and can take minutes, so refuse up front.
    f = img.astype(np.float32)
    if f.reshape(-1, 3).std(0).max() < 3.0 or np.abs(f[seed].mean(0) - f[mask == cv2.GC_BGD].mean(0)).max() < 6.0:
        return None
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)

    n, labels = cv2.connectedComponents(fg, connectivity=8)
    sy, sx = int(min(sh - 1, ey + 0.05 * short)), int(min(sw - 1, ex))
    lab = labels[sy, sx]
    if n <= 1 or lab == 0:
        return None
    obj = (labels == lab).astype(np.uint8)
    flood = obj.copy()  # fill holes: everything not reachable from the border is inside the object
    cv2.floodFill(flood, np.zeros((sh + 2, sw + 2), np.uint8), (0, 0), 1) if obj[0, 0] == 0 else None
    obj = obj | (1 - flood)
    area = obj.mean()
    if not 0.002 <= area <= 0.6:
        return None
    m = cv2.resize(obj.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    k = max(3, int(0.004 * min(w, h)) | 1)
    return np.clip(cv2.GaussianBlur(m, (k, k), 0), 0.0, 1.0)
