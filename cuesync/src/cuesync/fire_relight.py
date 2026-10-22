"""Local, exposure-aware light from a linear-RGB fire element.

Hand normals come from a landmark surface proxy. Background light uses a soft
distance field and optional projected hand blocker, not recovered room geometry.
The original image supplies every texture: no detail or hard shadow is invented.
All costly filters run in a cropped, resolution-capped work region.
"""
from __future__ import annotations

import cv2
import numpy as np

LUMA = np.array([.2126, .7152, .0722], np.float32)


def _smooth(a, b, x):
    y = np.clip((x-a)/(b-a), 0, 1)
    return y*y*(3-2*y)


def _hand_feather(matte, width):
    """Compute the full-resolution feather only where hand coverage exists."""
    x, y, w, h = cv2.boundingRect((matte > 0).astype(np.uint8))
    # Keep one background pixel of context when available. At the actual image
    # edge retain the original replicated border, including a full-frame hand.
    region = (slice(max(0, y-1), min(matte.shape[0], y+h+1)),
              slice(max(0, x-1), min(matte.shape[1], x+w+1)))
    if w == 0 or h == 0:
        return (slice(0, 0), slice(0, 0)), np.empty((0, 0), np.float32)
    local = matte[region]
    binary = cv2.copyMakeBorder((local > .05).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)[1:-1, 1:-1]
    return region, local*_smooth(0, width, np.maximum(inside-.5, 0))


def _material(P, matte, support, cfg, room_plate):
    h, w = matte.shape
    size = (max(1, w//4), max(1, h//4))
    low = cv2.resize(P, size, interpolation=cv2.INTER_AREA)
    low = cv2.bilateralFilter(low, 7, .028, 2.5)
    smooth = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
    Y = smooth@LUMA
    # Normalize background samples before filtering: bright skin cannot bleed
    # into the room material and produce an orange outline around the fingers.
    bg = 1-matte
    low_weight = cv2.resize(bg, size, interpolation=cv2.INTER_AREA)
    low_bg = cv2.resize(room_plate*bg[..., None], size, interpolation=cv2.INTER_AREA)
    low_bg /= np.maximum(low_weight[..., None], 1e-5)
    low_bg = cv2.bilateralFilter(low_bg, 7, .028, 2.5)
    smooth_bg = cv2.resize(low_bg, (w, h), interpolation=cv2.INTER_LINEAR)
    bg_Y = smooth_bg@LUMA
    near = support > .18
    hand_pixels = Y[(matte > .65) & near & (Y > .0015)]
    room_pixels = bg_Y[(matte < .05) & near & (bg_Y > .002)]
    hand_ambient = cfg.get('hand_ambient')
    room_ambient = cfg.get('room_ambient')
    if hand_ambient is None:
        hand_ambient = np.percentile(hand_pixels, 60) if hand_pixels.size > 20 else .12
    if room_ambient is None:
        room_ambient = np.percentile(room_pixels, 60) if room_pixels.size > 20 else .15
    hand_ambient = max(float(hand_ambient), .003)
    room_ambient = max(float(room_ambient), .004)
    hand_gain = min(max(.34/hand_ambient, 1/.70), float(cfg.get('max_hand_material_gain', 14.)))
    room_gain = min(.38/room_ambient, float(cfg.get('max_room_material_gain', 12.)))
    texture = 1+(np.clip((P@LUMA+.008)/(Y+.008), .75, 1.25)-1)*.45
    signal = _smooth(.0007, .0045, Y)
    hand = np.minimum(smooth*texture[..., None]*signal[..., None]*hand_gain, .92)
    texture = 1+(np.clip((room_plate@LUMA+.008)/(bg_Y+.008), .75, 1.25)-1)*.45
    signal = _smooth(.0007, .0045, bg_Y)
    room = np.minimum(smooth_bg*texture[..., None]*signal[..., None]*room_gain, .96)
    return hand, room, hand_ambient, room_ambient


def _room_visibility(matte, lum, ex, ey, p, scale, direction, strength, cfg):
    strength = float(np.clip(strength, 0, .65))
    if strength == 0:
        return np.ones_like(matte)
    rise = ((ex-p[0])*direction[0]+(ey-p[1])*direction[1])/scale
    lower = lum*((rise > -.05) & (rise < .65))
    weight = float(lower.sum())
    if weight < 1e-7:
        # A thrown source can leave the root bands entirely. Its blocker must
        # follow the actual flame, not keep projecting a shadow from the palm.
        weight = max(float(lum.sum()), 1e-7)
        sx, sy = float((ex*lum).sum()/weight), float((ey*lum).sum()/weight)
    else:
        sx = float((ex*lower).sum()/weight)
        sy = float((ey*lower).sum()/weight)
    projection = float(np.clip(cfg.get('room_shadow_scale', 1.38), 1.05, 1.7))
    M = np.array([[projection, 0, sx*(1-projection)],
                  [0, projection, sy*(1-projection)]], np.float32)
    h, w = matte.shape
    blocker = cv2.warpAffine(matte, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    blocker = cv2.GaussianBlur(blocker, (0, 0), max(float(cfg.get('room_shadow_blur', .14))*scale, 1.))
    return 1-strength*np.clip(blocker, 0, 1)


def _local(P, E, matte, p, scale, geo, z0, cfg, room_plate, direction):
    h, w = matte.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dx, dy = (xx-p[0])/scale, (yy-p[1])/scale
    along = dx*direction[0]+dy*direction[1]
    across = -dx*direction[1]+dy*direction[0]
    eq = cv2.resize(E, (max(1, w//8), max(1, h//8)), interpolation=cv2.INTER_AREA)
    ey, ex = np.mgrid[:eq.shape[0], :eq.shape[1]].astype(np.float32)
    qx, qy = w/eq.shape[1], h/eq.shape[0]
    ex, ey = (ex+.5)*qx-.5, (ey+.5)*qy-.5
    lum = eq@LUMA
    source_along = ((ex-p[0])*direction[0]+(ey-p[1])*direction[1])/scale
    source_sum = max(float(lum.sum()), 1e-9)
    total = source_sum*qx*qy/(scale*scale)
    center = np.array([(ex*lum).sum()/source_sum, (ey*lum).sum()/source_sum], np.float32)
    root_distance2 = ((ex-p[0])/scale)**2+((ey-p[1])/scale)**2
    root_energy = float((lum*np.exp(-.5*root_distance2/(.35*.35))).sum())
    contact_share = float(np.clip(root_energy/(source_sum*.08), 0, 1))
    diffuse = np.zeros((h, w), np.float32)
    confidence = np.zeros_like(matte)
    if geo is not None:
        n, depth = geo['normals'], geo['depth']
        confidence = np.clip(geo['light_valid']/np.maximum(matte, 1e-5), 0, 1)
    # A finite flame is represented by its actual lower, middle and upper
    # energy, rather than one stationary point light inside the palm.
    for lo, hi in ((-.04, .35), (.35, .8), (.8, 3.5)):
        band = lum*((source_along >= lo) & (source_along < hi))
        band_sum = float(band.sum())
        if band_sum < 1e-8:
            continue
        energy = band_sum*qx*qy/(scale*scale)
        sx, sy = float((ex*band).sum()/band_sum), float((ey*band).sum()/band_sum)
        source_height = (sx-p[0])*direction[0]+(sy-p[1])*direction[1]
        sz = z0-.24*scale-.22*source_height
        lx, ly = (sx-xx)/scale, (sy-yy)/scale
        # Segmentation can remain valid while landmarks briefly disappear, or
        # extend beyond the approximate capsules. Those pixels still reflect
        # firelight. Use a damped front-surface estimate where geometry is
        # uncertain, rather than treating a missing normal as black material.
        proxy_lz = (sz-z0)/scale
        proxy_d2 = lx*lx+ly*ly+proxy_lz*proxy_lz
        proxy_facing = np.clip(-proxy_lz/np.sqrt(proxy_d2+1e-6), 0, 1)
        proxy = .80*energy*proxy_facing/(proxy_d2+.15)
        if geo is None:
            diffuse += proxy
        else:
            lz = (sz-depth)/scale
            d2 = lx*lx+ly*ly+lz*lz
            facing = np.clip((n[..., 0]*lx+n[..., 1]*ly+n[..., 2]*lz)/np.sqrt(d2+1e-6), 0, 1)
            diffuse += confidence*energy*facing/(d2+.15)+(1-confidence)*proxy

    localized = np.exp(-.5*((dx/1.30)**2+(dy/1.15)**2))
    support = np.exp(-.5*((across/1.9)**2+((along-.35)/1.8)**2))
    hand_albedo, room_albedo, hand_ambient, room_ambient = _material(P, matte, support, cfg, room_plate)
    darkness = float(1-_smooth(.065, .29, room_ambient))
    dark_gain = float(cfg.get('dark_gain', 1+.3*darkness))
    light_gain = float(cfg.get('light', .48+1.82*darkness))
    contact_gain = float(cfg.get('contact', .19+.41*darkness))
    diffuse *= localized*light_gain
    diffuse = cv2.GaussianBlur(diffuse, (0, 0), max(.085*scale, 1))*matte
    hand_added = hand_albedo*np.array([1., .67, .35], np.float32)*diffuse[..., None]
    contact = np.exp(-.5*((dx/.29)**2+(dy/.21)**2))*matte
    hand_added += hand_albedo*np.array([1., .85, .61], np.float32)*contact[..., None]*contact_gain*min(total*2, 1.3)*contact_share
    # The wrist has no measured mesh: only a restrained, broad bounce is used.
    wrist = np.exp(-.5*((dx/.75)**2+((dy-.60)/.90)**2))*_smooth(-.1, .35, dy)*matte
    pulse = 1-np.exp(-2.6*total)
    hand_added += hand_albedo*np.array([1., .64, .32], np.float32)*wrist[..., None]*pulse*(.018+.030*darkness)*dark_gain

    distance = max(float(cfg.get('room_distance', 1.9)), 1.)
    room_dx, room_dy = (xx-center[0])/scale, (yy-center[1])/scale
    room_across = -room_dx*direction[1]+room_dy*direction[0]
    room_along = room_dx*direction[0]+room_dy*direction[1]
    r2 = (room_across/.95)**2+(room_along/1.1)**2
    room_field = (distance*distance/(r2+distance*distance))**1.5
    room_field *= np.exp(-.5*((room_across/1.6)**2+(room_along/1.7)**2))*(1-matte)
    occlusion = float(cfg.get('room_occlusion', .55*darkness))
    room_field *= _room_visibility(matte, lum, ex, ey, p, scale, direction, occlusion, cfg)
    strength = (.006+float(cfg.get('room_light', .22+.33*darkness))*darkness*dark_gain)*(1.9/distance)**2
    room_added = room_albedo*np.array([1., .61, .27], np.float32)*room_field[..., None]*pulse*strength
    palm = (matte > .5) & (contact > .15)
    room_measure = (matte < .05) & (room_field > .3)
    metrics = dict(source_power=round(total, 5),
                   palm_mean_added=float(hand_added[palm].mean()) if palm.any() else 0.,
                   room_mean_added=float(room_added[room_measure].mean()) if room_measure.any() else 0.,
                   hand_ambient=round(hand_ambient, 6), room_ambient=round(room_ambient, 6),
                   dark_response=round(darkness, 4), light_gain=light_gain, contact_gain=contact_gain,
                   room_distance=distance, room_occlusion=occlusion, geometry_valid=geo is not None,
                   geometry_fallback_share=float((1-confidence)[matte > .5].mean()) if np.any(matte > .5) else 0.,
                   contact_emission_share=contact_share,
                   light_direction=direction.tolist(),
                   room_geometry='soft diffuse field and projected SAM blocker; approximate room depth')
    return hand_added, room_added, metrics


def relight(P, E, matte, lm, p, scale, geo, z0, cfg=None):
    """Return ``(lit_rgb, added_rgb, metrics)`` without changing input arrays.

    RGB inputs are finite H x W x 3 linear images; E can exceed 1. Matte accepts
    coverage 0..1 or uint8 0..255. Missing matte/geometry degrades to room light.
    Missing/uncertain geometry with a valid matte uses a damped front-surface
    diffuse estimate, preserving illumination during brief landmark dropouts.
    Optional per-clip ``hand_ambient`` / ``room_ambient`` stabilize exposure.
    Other gains default to exposure-dependent values, with no clip identifiers.
    ``max_work_size`` defaults to 512 pixels on the cropped region's long edge.
    ``direction`` accepts a screen-space XY unit vector; its default is upward.
    ``hand_motion_px`` widens the lighting transition for blurred moving hands.
    """
    cfg = {} if cfg is None else cfg
    P = np.asarray(P, np.float32)
    E = np.asarray(E, np.float32)
    if P.ndim != 3 or P.shape[-1] != 3 or E.shape != P.shape:
        raise ValueError('Plate and emission must be matching H x W x 3 RGB arrays')
    if not np.isfinite(P).all() or not np.isfinite(E).all():
        raise ValueError('Plate and emission must contain finite values')
    h, w = P.shape[:2]
    point = np.asarray(p, np.float32)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError('Emitter point must contain two finite pixel coordinates')
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Hand scale must be finite and positive')
    direction = np.asarray(cfg.get('direction', [0, -1]), np.float32)
    if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-6:
        raise ValueError('Light direction must contain a finite nonzero XY vector')
    direction = direction/np.linalg.norm(direction)
    if matte is None:
        matte = np.zeros((h, w), np.float32)
    else:
        matte = np.asarray(matte, np.float32)
        if matte.shape != (h, w) or not np.isfinite(matte).all():
            raise ValueError('Matte must match the image dimensions and contain finite values')
        if matte.max(initial=0) > 1:
            matte = matte/255.
        matte = np.clip(matte, 0, 1)
    added = np.zeros_like(P)
    empty = dict(source_power=0., light_peak=0., palm_mean_added=0., room_mean_added=0.,
                 geometry_valid=False, work_pixels=0, roi=[0, 0, 0, 0])
    if E.max(initial=0) <= 1e-9:
        return P.copy(), added, empty

    # Truncate only the distant diffuse tail, with an interior edge feather.
    x0, x1 = max(0, int(np.floor(point[0]-4.5*scale))), min(w, int(np.ceil(point[0]+4.5*scale)))
    y0, y1 = max(0, int(np.floor(point[1]-4.5*scale))), min(h, int(np.ceil(point[1]+4.*scale)))
    if x1 <= x0 or y1 <= y0:
        return P.copy(), added, empty
    crop = (slice(y0, y1), slice(x0, x1))
    Pc, Ec, mc = P[crop], np.maximum(E[crop], 0), matte[crop]
    ch, cw = mc.shape
    max_size = max(64, int(cfg.get('max_work_size', 512)))
    ratio = min(1., max_size/max(ch, cw))
    size = (max(1, round(cw*ratio)), max(1, round(ch*ratio)))
    # Use one exact ratio for coordinates; rounding contributes less than a pixel.
    def resize(a, interpolation=cv2.INTER_AREA):
        return cv2.resize(a, size, interpolation=interpolation)
    low_mask = resize(mc)
    low_P, low_E = resize(Pc), resize(Ec)
    del Ec
    bg = 1-mc
    room_plate = resize(np.einsum('ijk,ij->ijk', Pc, bg))/np.maximum(resize(bg)[..., None], 1e-5)
    del bg
    low_geo = None
    if geo is not None:
        try:
            arrays = {key: np.asarray(geo[key], np.float32) for key in ('normals', 'depth', 'light_valid')}
        except (KeyError, TypeError) as error:
            raise ValueError('Geometry requires normals, depth and light_valid arrays') from error
        if arrays['normals'].shape != P.shape or any(arrays[k].shape != (h, w) for k in ('depth', 'light_valid')):
            raise ValueError('Geometry arrays must match plate dimensions')
        if all(np.isfinite(a).all() for a in arrays.values()) and arrays['light_valid'].max(initial=0) > 0:
            low_geo = {key: resize(a[crop]) for key, a in arrays.items()}
            low_geo['depth'] *= ratio
            low_geo['normals'] /= np.maximum(np.linalg.norm(low_geo['normals'], axis=-1, keepdims=True), 1e-5)
    depth0 = float(z0) if z0 is not None and np.isfinite(z0) else 0.
    local_point = (point-np.array([x0, y0], np.float32))*ratio
    hand, room, metrics = _local(low_P, low_E, low_mask, local_point, scale*ratio,
                                 low_geo, depth0*ratio, cfg, room_plate, direction)
    # Upsample illumination separately across the true full-resolution silhouette
    # so lowering the work resolution cannot light the black gaps between fingers.
    hand /= np.maximum(low_mask[..., None], 1e-5)
    room /= np.maximum(1-low_mask[..., None], 1e-5)
    # The scalar room pulse can promote this small work map to float64. Keep
    # reconstruction in the same float32 precision as the public RGB outputs.
    room = room.astype(np.float32, copy=False)
    # SAM can be sharp while the photographed hand is motion-blurred. Feather
    # added hand light just inside its silhouette, rather than painting a hard
    # bright polygon or spilling skin-colored light into background finger gaps.
    motion = max(float(cfg.get('hand_motion_px', 0.)), 0.)
    edge_width = max(1.5, .022*scale, min(.6*motion, .12*scale))
    hand_region, light_mask = _hand_feather(mc, edge_width)
    reflection = cv2.resize(room, (cw, ch), interpolation=cv2.INTER_LINEAR)
    # Channel-wise multiplication avoids NumPy's slow three-element broadcast
    # loop and keeps one reusable RGB scratch instead of several 1080p copies.
    scratch = np.empty_like(reflection)
    if light_mask.size:
        local_bg = 1-mc[hand_region]
        cv2.resize(hand, (cw, ch), dst=scratch, interpolation=cv2.INTER_LINEAR)
        for c in range(3):
            reflection[hand_region][..., c] *= local_bg
            scratch[hand_region][..., c] *= light_mask
        reflection[hand_region] += scratch[hand_region]
        # The complementary room contribution must cover the inward feather too:
        # otherwise both lights vanish in a strip and trace a dark SAM outline.
        # Extend only background-derived room radiance into this narrow transition;
        # never blur skin material out into the background or the finger gaps.
        sigma = max(.65*edge_width*ratio, 1.)
        room_weight = 1-low_mask
        extended = cv2.GaussianBlur(room*room_weight[..., None], (0, 0), sigma)
        extended /= np.maximum(cv2.GaussianBlur(room_weight, (0, 0), sigma)[..., None], 1e-5)
        cv2.resize(extended, (cw, ch), dst=scratch, interpolation=cv2.INTER_LINEAR)
        transition = np.subtract(mc[hand_region], light_mask, out=light_mask)
        for c in range(3):
            scratch[hand_region][..., c] *= transition
        reflection[hand_region] += scratch[hand_region]
        del transition, local_bg
    del light_mask
    feather = max(.35*scale, 1.)
    wx, wy = np.ones(cw, np.float32), np.ones(ch, np.float32)
    if x0 > 0: wx *= _smooth(0, feather, np.arange(cw))
    if x1 < w: wx *= _smooth(0, feather, np.arange(cw)[::-1])
    if y0 > 0: wy *= _smooth(0, feather, np.arange(ch))
    if y1 < h: wy *= _smooth(0, feather, np.arange(ch)[::-1])
    edge_fade = wy[:, None]*wx[None, :]
    for c in range(3):
        reflection[..., c] *= edge_fade
    del edge_fade
    headroom = np.maximum(1-Pc, 0)
    np.maximum(headroom, 1e-7, out=scratch)
    np.maximum(reflection, 0, out=reflection)
    np.divide(reflection, scratch, out=reflection)
    np.tanh(reflection, out=reflection)
    reflection *= headroom
    del scratch, headroom
    added[crop] = reflection
    metrics.update(light_peak=float(reflection.max(initial=0)), work_pixels=int(size[0]*size[1]),
                   roi=[x0, y0, x1, y1], work_scale=ratio)
    return P+added, added, metrics


light_hand = relight
