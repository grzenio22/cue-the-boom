"""Conservative hand geometry for fire occlusion and directional lighting.

Landmark XY and MediaPipe Z use the same pixel scale. Smaller Z is nearer
the camera; normals face the camera (-Z). This is an approximate surface,
not a metric reconstruction. Segmentation remains the final support.
"""
from __future__ import annotations

import cv2
import numpy as np

PALM = (0, 1, 5, 9, 13, 17)
FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
           (13, 14, 15, 16), (17, 18, 19, 20))
_DISTAL = frozenset((3, 4, 7, 8, 11, 12, 15, 16, 19, 20))


def _points(landmarks):
    if landmarks is None:
        return None
    p = np.asarray(landmarks, dtype=np.float32)
    if p.shape != (21, 3):
        raise ValueError("landmarks must have shape (21, 3) in pixel coordinates")
    return p if np.isfinite(p).all() else None


def _position(point):
    if point is None:
        return None
    q = np.asarray(point, dtype=np.float32)
    if q.ndim != 1 or len(q) < 2 or not np.isfinite(q[:2]).all():
        return None
    return q[:2]


def _triangles(p):
    ring = p[list(PALM)]
    center = ring.mean(axis=0)
    return [(center, ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]


def _bary(a, b, c, x, y):
    den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(float(den)) < 1e-5:
        return None
    u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / den
    v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / den
    return u, v, 1 - u - v


def emitter_depth(landmarks, point):
    """Interpolate palm Z, projecting to its nearest edge outside the palm.

    Missing tracking or source position returns zero. Any intended distance
    above the skin belongs to the compositor, rather than this estimate.
    """
    p, q = _points(landmarks), _position(point)
    if p is None or q is None:
        return 0.0
    best_distance, best_z = np.inf, float(p[9, 2])
    for a, b, c in _triangles(p):
        uvw = _bary(a, b, c, q[0], q[1])
        if uvw is not None and min(uvw) >= -1e-5:
            return float(uvw[0] * a[2] + uvw[1] * b[2] + uvw[2] * c[2])
        for v0, v1 in ((a, b), (b, c), (c, a)):
            d = v1[:2] - v0[:2]
            t = np.clip(np.dot(q - v0[:2], d) / max(float(d @ d), 1e-6), 0, 1)
            distance = float(np.sum((q - (v0[:2] + t * d)) ** 2))
            if distance < best_distance:
                best_distance = distance
                best_z = float(v0[2] + t * (v1[2] - v0[2]))
    return best_z


def emission_depth(landmarks, point):
    """Use local distal-finger depth for sources attached near a fingertip."""
    p, q = _points(landmarks), _position(point)
    if p is None or q is None:
        return 0.0
    distances = np.linalg.norm(p[:, :2] - q, axis=1)
    joint = int(np.argmin(distances))
    palm_size = float(np.linalg.norm(p[0, :2] - p[9, :2]))
    if joint in _DISTAL and distances[joint] < 0.12 * palm_size:
        return float(p[joint, 2])
    return emitter_depth(p, q)


def _local_surface(p, mask, scale):
    """Evaluate only the cropped hand, in a common isotropic pixel scale."""
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    z = np.full((h, w), np.inf, np.float32)
    support = np.zeros((h, w), np.float32)
    for a, b, c in _triangles(p):
        uvw = _bary(a, b, c, xx, yy)
        if uvw is None:
            continue
        u, v, t = uvw
        inside = (u >= 0) & (v >= 0) & (t >= 0)
        zi = u * a[2] + v * b[2] + t * c[2]
        z[inside] = np.minimum(z[inside], zi[inside])
        support[inside] = 1

    # The crop's zero border makes all-foreground mattes finite too. Radius
    # bounds protect palm-connected MCPs from becoming oversized spheres.
    binary = np.pad((mask > 0.25).astype(np.uint8), 1)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)[1:-1, 1:-1]

    def radius_at(point):
        x, y = np.clip(point[:2], [0, 0], [w - 1, h - 1])
        ix, iy = int(x), int(y)
        jx, jy = min(ix + 1, w - 1), min(iy + 1, h - 1)
        tx, ty = x - ix, y - iy
        d = ((1 - tx) * (1 - ty) * distance[iy, ix] + tx * (1 - ty) * distance[iy, jx]
             + (1 - tx) * ty * distance[jy, ix] + tx * ty * distance[jy, jx])
        return float(np.clip(d * 1.08, 0.04 * scale, 0.14 * scale))

    for ids in FINGERS:
        for joint in range(3):
            a, b = p[ids[joint]], p[ids[joint + 1]]
            dx, dy = b[0] - a[0], b[1] - a[1]
            t = np.clip(((xx - a[0]) * dx + (yy - a[1]) * dy)
                        / max(float(dx * dx + dy * dy), 1e-6), 0, 1)
            dist = np.hypot(xx - (a[0] + t * dx), yy - (a[1] + t * dy))
            r0, rm, r1 = radius_at(a), radius_at((a + b) / 2), radius_at(b)
            radius = np.where(t < 0.5, r0 + (rm - r0) * (2 * t), rm + (r1 - rm) * (2 * t - 1))
            r = dist / np.maximum(radius, 1)
            inside = r < 1
            zi = a[2] + t * (b[2] - a[2]) - 0.5 * radius * np.sqrt(np.maximum(1 - r * r, 0))
            z[inside] = np.minimum(z[inside], zi[inside])
            support = np.maximum(support, np.clip((1 - r) * radius / 1.2, 0, 1))

    finite = np.isfinite(z)
    z[~finite] = 0
    sigma = max(0.8, min(2, 0.008 * scale))
    weight = finite.astype(np.float32)
    blur_w = cv2.GaussianBlur(weight, (0, 0), sigma)
    smooth_z = cv2.GaussianBlur(z * weight, (0, 0), sigma) / np.maximum(blur_w, 1e-5)
    dzdy = np.gradient(smooth_z, axis=0) if h > 1 else np.zeros_like(smooth_z)
    dzdx = np.gradient(smooth_z, axis=1) if w > 1 else np.zeros_like(smooth_z)
    normals = np.stack((np.clip(dzdx, -3, 3), np.clip(dzdy, -3, 3), -np.ones_like(z)), axis=-1)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-5)
    coverage = support * mask

    # Extend illumination smoothly across small geometry gaps inside the real
    # hand silhouette. Occlusion uses the stricter coverage instead.
    light_sigma = max(1, 0.05 * scale)
    light_w = cv2.GaussianBlur(coverage, (0, 0), light_sigma)
    light_z = cv2.GaussianBlur(smooth_z * coverage, (0, 0), light_sigma) / np.maximum(light_w, 1e-5)
    light_n = (cv2.GaussianBlur(normals * coverage[..., None], (0, 0), light_sigma)
               / np.maximum(light_w[..., None], 1e-5))
    light_n /= np.maximum(np.linalg.norm(light_n, axis=-1, keepdims=True), 1e-5)
    light_coverage = light_w * mask
    light_coverage[light_w < 1e-4] = 0
    in_surface = coverage > 0.05
    return dict(
        depth=np.where(in_surface, smooth_z, np.where(light_coverage > 0, light_z, 0)),
        normals=np.where(in_surface[..., None], normals, light_n),
        valid=coverage,
        light_valid=light_coverage,
    )


def surface(landmarks, matte, scale):
    """Return full-image depth, normals, valid, light_valid and an XYXY bbox.

    Palm triangles interpolate Z, and tapered finger capsules approximate the
    convex front of each finger. The nearest supported surface wins overlaps.
    Missing/nonfinite landmarks return zero geometry. Mattes accept normalized
    coverage or byte coverage. Malformed input shapes and invalid scale raise
    ValueError. The input arrays are never modified.

    Expensive geometry runs only in the hand crop. Hands larger than 200 pixels
    use an isotropically reduced grid; the original matte is reapplied after
    upsampling, so even one-pixel segmentation gaps remain unoccluded/unlit.
    """
    mask = np.asarray(matte, dtype=np.float32)
    if mask.ndim != 2 or not np.isfinite(mask).all():
        raise ValueError("matte must be a finite two-dimensional coverage map")
    s = float(scale)
    if not np.isfinite(s) or s <= 0:
        raise ValueError("scale must be finite and positive")
    p = _points(landmarks)
    h, w = mask.shape
    result = dict(depth=np.zeros((h, w), np.float32),
                  normals=np.zeros((h, w, 3), np.float32),
                  valid=np.zeros((h, w), np.float32),
                  light_valid=np.zeros((h, w), np.float32), bbox=(0, 0, 0, 0))
    if p is None or h == 0 or w == 0:
        return result
    if mask.max(initial=0) > 1:
        mask = mask / 255
    mask = np.clip(mask, 0, 1)
    s = max(s, 4)
    pad = int(np.ceil(0.32 * s + 4))
    x0 = max(0, int(np.floor(p[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(p[:, 1].min())) - pad)
    x1 = min(w, int(np.ceil(p[:, 0].max())) + pad + 1)
    y1 = min(h, int(np.ceil(p[:, 1].max())) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return result
    crop = mask[y0:y1, x0:x1]
    if not np.any(crop > 0):
        return result
    result["bbox"] = (x0, y0, x1, y1)
    local_p = p.copy()
    local_p[:, :2] -= (x0, y0)
    q = max(1, int(np.ceil(s / 200)))
    ch, cw = crop.shape
    ph, pw = ((ch + q - 1) // q) * q, ((cw + q - 1) // q) * q
    if q > 1:
        # Padding keeps XY and Z in the same scale even for odd-sized crops.
        padded = np.pad(crop, ((0, ph - ch), (0, pw - cw)))
        low_mask = cv2.resize(padded, (pw // q, ph // q), interpolation=cv2.INTER_AREA)
        local_p[:, :2] = (local_p[:, :2] + 0.5) / q - 0.5
        local_p[:, 2] /= q
    else:
        low_mask = crop
    geo = _local_surface(local_p, low_mask, s / q)
    for key, value in geo.items():
        if q > 1:
            value = cv2.resize(value, (pw, ph), interpolation=cv2.INTER_LINEAR)[:ch, :cw]
            if key == "depth":
                value *= q
            elif key == "normals":
                value /= np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-5)
        if key in ("valid", "light_valid"):
            value = np.minimum(value, crop)
        result[key][y0:y1, x0:x1] = value
    return result
