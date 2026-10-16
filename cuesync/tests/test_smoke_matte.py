"""Smoke over bright plates keeps its shading and never draws contours; material transitions stay opaque."""

import numpy as np

from cuesync.composite import CompositeParams, composite_frame


def cloud(h=512, w=288, rim=190.0, centre=110.0, radius=0.32):
    """A smooth round cloud on black: bright rim, self-shadowed darker centre, soft edge."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot((xx - w / 2) / w, (yy - h / 2) / w) / radius  # 0 centre .. 1 edge
    body = np.clip(1.0 - (d - 0.85) / 0.15, 0.0, 1.0)  # soft edge
    luma = (centre + (rim - centre) * np.clip(d, 0, 1) ** 2) * body
    return np.repeat(luma[..., None], 3, -1).clip(0, 255).astype(np.uint8), d


def comp(plate_level, elem, **kw):
    plate = np.full(elem.shape, plate_level, np.uint8)
    return composite_frame(plate, elem, CompositeParams(grain=False), fire_effect=False, dense_smoke=True,
                           smoke_ref=190 / 255, black_floor=0, **kw).astype(np.float32)


def test_dense_smoke_keeps_its_shading_on_a_bright_plate():
    elem, d = cloud()
    out = comp(225, elem)[..., 1]
    centre = out[d < 0.2].mean()
    rim = out[(d > 0.7) & (d < 0.8)].mean()
    assert centre < 225 - 40  # the self-shadowed centre reads as dense grey smoke, not a milky veil
    assert centre < rim - 20  # and darker than the lit rim: the volume survives


def test_dense_smoke_hides_most_of_the_plate_behind_it():
    """A striped bright wall behind the cloud: inside its core the stripes all but disappear."""
    elem, d = cloud()
    plate = np.full(elem.shape, 180, np.uint8)
    plate[:, ::8] = 230
    out = composite_frame(plate, elem, CompositeParams(grain=False), fire_effect=False, dense_smoke=True,
                          smoke_ref=190 / 255, black_floor=0).astype(np.float32)[..., 1]
    core = d < 0.4
    cols = np.zeros(elem.shape[:2], bool)
    cols[:, ::8] = True
    contrast = out[core & cols].mean() - out[core & ~cols].mean()
    assert contrast < 0.15 * (230 - 180)


def test_a_smooth_cloud_composites_without_contours():
    elem, d = cloud()
    out = comp(225, elem)[..., 1]
    line = out[elem.shape[0] // 2, :]
    inside = d[elem.shape[0] // 2, :] < 0.8
    steps = np.abs(np.diff(line))[inside[1:] & inside[:-1]]
    assert steps.max() <= 4  # no luma-contour bands: the source steps by <=2 levels per pixel here


def test_a_fire_to_smoke_transition_is_never_more_see_through_than_its_sides():
    h, w = 64, 256
    elem = np.zeros((h, w, 3), np.uint8)
    t = np.linspace(0, 1, w, dtype=np.float32)
    fire = np.array([255, 150, 40], np.float32)
    smoke = np.array([200, 200, 200], np.float32)
    elem[:] = ((1 - t)[:, None] * fire + t[:, None] * smoke)[None].astype(np.uint8)
    plate = np.full((h, w, 3), 30, np.uint8)
    p2 = np.full((h, w, 3), 220, np.uint8)
    kw = dict(fire_effect=True, dense_smoke=False, smoke_ref=200 / 255, black_floor=0)
    a = composite_frame(plate, elem, CompositeParams(grain=False, bloom_gain=0.0), **kw).astype(np.float32)
    b = composite_frame(p2, elem, CompositeParams(grain=False, bloom_gain=0.0), **kw).astype(np.float32)
    transmission = ((b - a)[h // 2].mean(-1)) / (220 - 30)
    ends = max(transmission[:16].mean(), transmission[-16:].mean())
    assert transmission.max() <= ends + 0.08
