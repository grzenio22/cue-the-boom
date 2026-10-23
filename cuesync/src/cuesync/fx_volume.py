"""A finite-thickness reconstruction of an existing fire image, on the CPU.

This is an art-directed approximate volume, NOT recovered 3D geometry or a
physical fire simulation. Nine correlated, advecting depth slabs share the
source's energy. Small depth-dependent shifts and Beer--Lambert integration
produce internal overlap without stacking nine copies of the flame.

Input/output RGB is scene-linear float32. ``t`` is time in seconds; ``view`` is
a small horizontal viewpoint proxy, normally -0.15..+0.15. Returned transmission
multiplies the plate BEFORE adding emission. Never apply another alpha to the
emission. Black source pixels stay black except inside the small parallax halo.
"""
from __future__ import annotations

import cv2
import numpy as np

LUMA = np.array([.2126, .7152, .0722], np.float32)
DEFAULTS = dict(slabs=9, thickness=.9, parallax=.60, flow=.040,
                optical_depth=.85, tau_cap=1.25, density_contrast=.34,
                foreground_cooling=.34, filament_relief=.32, energy=.95)


def _smooth(a, b, x):
    u = np.clip((x-a)/(b-a), 0., 1.)
    return u*u*(3.-2.*u)


def _prepare_noise(x, y, t):
    """Cache the depth-invariant phases of the advecting 3D field."""
    yy = y + t*.40
    a = x*5.3+yy*3.6
    b = x*10.1-yy*7.9
    c = x*18.4+yy*11.2
    d = x*34.7-yy*20.5

    def evaluate(z):
        # Keep the original operation order, including the time phase terms;
        # reassociating float32 sums can alter subpixel remapping downstream.
        return (np.sin(a+z*3.1) * .25
                + np.sin(b+z*5.7+t*.24) * .18
                + np.sin(c-z*8.8-t*.19) * .11
                + np.sin(d+z*13.4) * .055)

    return evaluate


def volume_element(E, t, view=0.0, config=None, *, active_bounds=None, crop_output=False):
    """Return (emission, transmission, numeric diagnostics), source-sized.

    Cropping internally keeps work proportional to flame extent. RGB emission
    remains bounded to source energy (``energy`` may only attenuate). Transmission
    represents intrinsic optical thickness; the final compositor must bound it
    against the actual background. Only intrinsically faint gas is thinned here.
    ``view`` may be scalar or a two-value vector (horizontal component used).

    ``active_bounds`` is an optional trusted XYXY box containing ALL nonzero
    source pixels. It avoids scanning and cleaning a mostly black full canvas.
    The original canvas still determines the padded volume crop and parallax.
    With ``crop_output=True``, arrays cover only ``diagnostics['crop']`` (XYXY),
    rather than allocating full-size black RGB and unit transmission images.
    An empty source then returns 0 x 0 arrays and the box [0, 0, 0, 0].
    """
    cfg = DEFAULTS | (config or {})
    E = np.asarray(E, np.float32)
    if E.ndim != 3 or E.shape[2] != 3:
        raise ValueError('Expected H x W x 3 linear RGB emission')
    n = int(cfg['slabs'])
    if not 7 <= n <= 11:
        raise ValueError('Use 7 to 11 depth slabs')
    view = float(np.asarray(view).flat[0])
    view = float(np.clip(view, -.3, .3))
    h0, w0 = E.shape[:2]
    bx, by, bx1, by1 = 0, 0, w0, h0
    if active_bounds is not None:
        bounds = np.asarray(active_bounds)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError('active_bounds must be finite XYXY pixel coordinates')
        bx, by, bx1, by1 = (int(v) for v in bounds)
        bx, by = int(np.clip(bx, 0, w0)), int(np.clip(by, 0, h0))
        bx1, by1 = int(np.clip(bx1, bx, w0)), int(np.clip(by1, by, h0))
    clean = np.maximum(np.nan_to_num(E[by:by1, bx:bx1], nan=0., posinf=0., neginf=0.), 0.)
    lum0 = clean @ LUMA
    src_energy = float(lum0.sum(dtype=np.float64))
    peak = float(lum0.max(initial=0.))
    ys, xs = np.nonzero(lum0 > peak*.001)
    if len(xs) == 0 or src_energy <= 0.:
        shape = (0, 0) if crop_output else (h0, w0)
        return np.zeros((*shape, 3), np.float32), np.ones(shape, np.float32), dict(
                                           slabs=n, energy_ratio=0., opacity_peak=0., crop=[0, 0, 0, 0],
                                           method='approximate reconstructed volume')
    pad = max(5, int((xs.max()-xs.min())*.055))
    x0 = max(0, bx+int(xs.min())-pad); x1 = min(w0, bx+int(xs.max())+pad+1)
    y0 = max(0, by+int(ys.min())-pad); y1 = min(h0, by+int(ys.max())+pad+1)
    if active_bounds is None:
        src = clean[y0:y1, x0:x1]
    else:
        src = np.maximum(np.nan_to_num(E[y0:y1, x0:x1], nan=0., posinf=0., neginf=0.), 0.)
    lum = src @ LUMA
    h, w = lum.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)

    # Coherent per-row cross sections, fitted to source energy and smoothed over
    # the height. No independently wobbling card edges or duplicated silhouettes.
    rowmass = lum.sum(1)
    center = (lum*xx).sum(1)/np.maximum(rowmass, 1e-8)
    valid = np.flatnonzero(rowmass > src_energy*1e-6)
    center = np.interp(np.arange(h), valid, center[valid]).astype(np.float32)
    sigma = np.sqrt((lum*(xx-center[:, None])**2).sum(1)/np.maximum(rowmass, 1e-8))
    sigma = np.interp(np.arange(h), valid, sigma[valid]).astype(np.float32)
    smooth_y = max(h*.025, 1.)
    center = cv2.GaussianBlur(center[:, None], (1, 0), smooth_y).ravel()
    radius = cv2.GaussianBlur(sigma[:, None], (1, 0), smooth_y).ravel()*2.2
    radius = np.maximum(radius, max(w*.035, 2.))
    xn = (xx-center[:, None])/radius[:, None]
    yn = yy/max(h, 1)*3.
    support = _smooth(.0001, .003, lum)
    # Place the source's existing bright filaments slightly nearer the viewer
    # and its valleys behind them. No new texture or white core is invented.
    local_lum = cv2.GaussianBlur(lum, (0, 0), max(w*.012, 1.))
    relief = np.clip((lum+.004)/(local_lum+.004), .55, 1.65)
    depth_bias = (relief-1.)*cfg['filament_relief']
    # Keep cross sections broad: a noisy source silhouette remains authoritative.
    thickness = np.sqrt(np.clip(1.-(xn/1.55)**2, .025, 1.))*cfg['thickness']
    zs = np.linspace(-1., 1., n, dtype=np.float32)
    noise_field = _prepare_noise(xn, yn, t)
    thickness_denom = np.maximum(thickness, .18)
    fields = []
    for z in zs:
        field = np.maximum(.06, 1.+noise_field(float(z)*1.8)*1.35)
        envelope = np.exp(-1.8*((z-depth_bias)/thickness_denom)**2)
        fields.append(field*envelope)
    weights = np.stack(fields).astype(np.float32)
    weights /= np.maximum(weights.sum(0), 1e-8)

    out = np.zeros_like(src)
    trans = np.ones((h, w), np.float32)
    # Broad filaments are denser, while thin orange rims retain transparency.
    positive = lum[lum > float(lum.max())*.003]
    normlum = lum/max(float(np.percentile(positive, 94)), .02)
    # Reallocate a little low-density orange veil into interior filaments. This
    # does not increase total light energy and avoids a uniform planar orange
    # wash over bright backgrounds. The source still controls all large forms.
    density_emission = np.clip(normlum, .015, 1.8)**cfg['density_contrast']
    tau = cfg['optical_depth']*np.clip(normlum, 0., 1.8)**.70*support
    tau *= .78+.22*relief
    tau = np.minimum(tau, cfg['tau_cap'])
    hot = _smooth(.20, .85, normlum)
    motion_weight = _smooth(0., .12, yy/max(h, 1))*(1.-_smooth(.88, 1., yy/max(h, 1)))
    amber = np.array([1., .39, .075], np.float32)
    target = (lum/(amber@LUMA))[..., None]*amber
    eddy_x, eddy_y = xn*.72, yn*.70
    eddy_field = _prepare_noise(eddy_x, eddy_y, t)
    cooling_base = .35+.65*(1-hot)
    cooling_relief = np.clip(relief, .7, 1.3)
    # Integrate back to front. A small thermal color change varies within volume;
    # light travels through cooler front sheets instead of opaque orange decals.
    for i, z in enumerate(zs):
        eddy = eddy_field(float(z)*1.4)
        shift = z*radius[:, None]*(view*cfg['parallax'] + cfg['flow']*eddy)*motion_weight
        mapx = (xx-shift).astype(np.float32)
        mapy = (yy-z*cfg['flow']*radius[:, None]*eddy*.22*motion_weight).astype(np.float32)
        wt = weights[i]
        layer = src*wt[..., None]
        front = float((z+1.)*.5)
        cooling = cfg['foreground_cooling']*front**1.4*cooling_base
        cooling *= cooling_relief
        # Cooler foreground ridges preserve their luminance; the unmodified
        # source supplies the hot interior instead of a synthesized pale core.
        layer = layer*(1.-cooling[..., None])+target*wt[..., None]*cooling[..., None]
        layer *= (density_emission*(1.+eddy*.16))[..., None]
        layer = cv2.remap(layer, mapx, mapy, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
        layer_tau = cv2.remap(tau*wt, mapx, mapy, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        layer_trans = np.exp(-layer_tau)
        out = out*layer_trans[..., None] + layer
        trans *= layer_trans
    target_energy = src_energy*float(np.clip(cfg['energy'], 0., 1.))
    out_energy = float((out@LUMA).sum(dtype=np.float64))
    if out_energy > 0.:
        out *= target_energy/out_energy
    # Preserve body thickness until the actual background is known. An assumed
    # white background made almost every bright-wall body pixel transparent.
    # Absolute faintness still removes extinction, including energy attenuation
    # to zero; normalized source density must not leave opaque invisible gas.
    opacity = (1.-trans)*_smooth(.0001, .003, out@LUMA)
    trans = 1.-opacity
    diagnostics = dict(slabs=n, energy_ratio=float((out@LUMA).sum(dtype=np.float64)/src_energy),
                       opacity_peak=float(opacity.max()), crop=[x0,y0,x1,y1],
                       view=view, max_parallax_px=float(np.max(radius)*abs(view)*cfg['parallax']),
                       tau_peak=float(tau.max()),
                       method='approximate reconstructed volume; source-derived relief and cooler foreground slabs')
    if crop_output:
        return out, trans, diagnostics
    emission = np.zeros_like(E)
    transmission = np.ones((h0, w0), np.float32)
    emission[y0:y1, x0:x1] = out
    transmission[y0:y1, x0:x1] = trans
    return emission, transmission, diagnostics
