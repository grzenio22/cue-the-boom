"""Authored source calibration stays consistent across CPU replay and live preview."""
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cuesync import bending as b
from cuesync.media import read_video, write_video
from cuesync.pipeline import ShotSpec
from cuesync.server import create_app


@pytest.fixture
def saved_fire(tmp_path, make_video):
    n, w, h = 48, 128, 128
    plate = make_video(tmp_path, n=n, fps=24, w=w, h=h, color_fn=lambda _: (40, 35, 30))
    track = {"version": 1, "fps": 24, "W": w, "H": h, "emitter": "palm", "hand": "left",
             "xy": [[64., 100.]] * n, "dir": [[0., -1.]] * n, "scale": [20.] * n,
             "conf": [1.] * n, "src": ["mp"] * n, "speed": [0.] * n, "landmarks": [None] * n,
             "gesture": ["Open_Palm"] * n, "events": []}
    source = tmp_path / "library" / "fire_burst"
    source.mkdir(parents=True)
    track_path = source / "hand_track.json"
    track_path.write_text(json.dumps(track))
    frames = np.zeros((n, h, w, 3), np.uint8)
    frames[4:35, 60:94, 52:77] = [180, 90, 18]
    frames[35:39, 65:91, 56:73] = [100, 35, 6]
    write_video(source / "element.mp4", frames, fps=24)
    receipt = {"effect": {"id": "fire_burst", "behaviour": "burst", "fluid": "fire"},
               "render": {"prompt": "synthetic cached fire fixture", "guide_scale": 1.},
               "bending": {"ref": {"xy": [64., 100.], "dir": [0., -1.], "scale": 20.}, "thrown": None}}
    (source / "receipt.json").write_text(json.dumps(receipt))
    spec = ShotSpec(plate=plate, voice=None, effect="fire_burst", point=(.5, 100 / h), cue="hold",
                    holds=[(.5, 1.)], plate_size=(w, h), max_plate_s=2.)
    return source, track_path, spec, receipt


def deny_render(*args, **kwargs):
    raise AssertionError("A source calibration/replay must never generate a new element")


def set_profile(source, receipt, profile):
    (source / "receipt.json").write_text(json.dumps(receipt | {"source_profile": profile}))


def test_missing_source_profile_has_identity_defaults_and_partial_profiles_are_normalized():
    assert b.source_profile(None) == {"scale": 1., "exposure": 1.}
    assert b.source_profile({}) == {"scale": 1., "exposure": 1.}
    assert b.source_profile({"scale": .5}) == {"scale": .5, "exposure": 1.}
    assert b.source_profile({"exposure": 2}) == {"scale": 1., "exposure": 2.}


@pytest.mark.parametrize("profile", [[], "bad", {"scale": 0}, {"scale": 2.01}, {"scale": .24},
                                      {"scale": float("nan")}, {"scale": True}, {"scale": "1"},
                                      {"exposure": .24}, {"exposure": 4.01}, {"exposure": float("inf")},
                                      {"exposure": None}, {"gain": 1.},
                                      pytest.param({"scale": 10 ** 400}, id="oversized_integer")])
def test_invalid_source_profile_is_rejected_clearly(profile):
    with pytest.raises(ValueError, match="source_profile"):
        b.source_profile(profile)


@pytest.mark.parametrize("exposure", [.5, 1., 2.])
def test_source_exposure_changes_emission_and_optical_depth_before_warp_and_relight(
    saved_fire, tmp_path, monkeypatch, exposure
):
    source, track, spec, receipt = saved_fire
    warped, lights = [], []
    warp, relight = b.warp_element, b.fire_light.relight

    def observe_warp(E, *args, **kwargs):
        warped.append(E.copy())
        return warp(E, *args, **kwargs)

    def observe_light(P, E, *args, **kwargs):
        lights.append(E.copy())
        return relight(P, E, *args, **kwargs)

    monkeypatch.setattr(b, "warp_element", observe_warp)
    monkeypatch.setattr(b.fire_light, "relight", observe_light)
    b.make_bending_shot(spec, deny_render, tmp_path / "identity", track, element_from=source)
    original_warps, original_lights = warped[:], lights[:]
    warped.clear()
    lights.clear()
    set_profile(source, receipt, {"exposure": exposure})
    result = b.make_bending_shot(spec, deny_render, tmp_path / "exposed", track, element_from=source)
    assert len(warped) == len(original_warps) > 5
    assert max(float(E[..., :3].max()) for E in original_warps) > .1
    for before, after in zip(original_warps, warped):
        np.testing.assert_allclose(after[..., :3], before[..., :3] * exposure, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(after[..., 3], 1. - (1. - before[..., 3]) ** exposure, atol=1e-7)
    assert len(lights) == len(original_lights) > 5
    for before, after in zip(original_lights, lights):
        np.testing.assert_allclose(after, before * exposure, rtol=2e-6, atol=1e-7)
    assert result.receipt["source_profile"] == {"scale": 1., "exposure": exposure}
    assert result.receipt["render"]["seconds"] == 0.


@pytest.mark.parametrize("effect,behaviour", [("fire_burst", "burst"), ("flame_plume", "plume")])
def test_source_scale_is_spatial_only_and_survives_perform_again(saved_fire, tmp_path, monkeypatch, effect, behaviour):
    source, track, spec, receipt = saved_fire
    receipt["effect"] = {"id": effect, "behaviour": behaviour, "fluid": "fire"}
    (source / "receipt.json").write_text(json.dumps(receipt))
    scales = []
    warp = b.warp_element

    def observe(E, track, ref, t, lag, size=1., **kwargs):
        scales.append(size)
        return warp(E, track, ref, t, lag, size, **kwargs)

    monkeypatch.setattr(b, "warp_element", observe)
    original = b.make_bending_shot(spec, deny_render, tmp_path / "original", track, element_from=source)
    original_scales = scales[:]
    scales.clear()
    set_profile(source, receipt, {"scale": .5, "exposure": 1.})
    calibrated = b.make_bending_shot(spec, deny_render, tmp_path / "calibrated", track, element_from=source)
    assert len(scales) == len(original_scales) > 5
    np.testing.assert_allclose(scales, np.asarray(original_scales) * .5)
    assert calibrated.receipt["bending"]["hits"] == original.receipt["bending"]["hits"]
    assert calibrated.receipt["bending"]["fit"]["play"] == pytest.approx(
        original.receipt["bending"]["fit"]["play"] * .5, abs=.01)
    again = b.bending_again(calibrated.out_dir, spec, tmp_path / "again")
    assert again.receipt["source_profile"] == {"scale": .5, "exposure": 1.}
    assert again.receipt["render"]["seconds"] == 0.
    assert again.receipt["bending"]["performed_again"]["no_new_render"]
    np.testing.assert_array_equal(read_video(again.composite).frames, read_video(calibrated.composite).frames)


def test_library_preview_fit_includes_source_scale_once_and_exposes_calibration(saved_fire, tmp_path, monkeypatch):
    source, _, _, receipt = saved_fire
    monkeypatch.setattr(b, "library_dir", lambda: source.parent)
    original = TestClient(create_app(None, tmp_path / "before-jobs")).get("/api/library").json()[0]
    set_profile(source, receipt, {"scale": .5, "exposure": 2.})
    client = TestClient(create_app(None, tmp_path / "after-jobs"))
    calibrated = client.get("/api/library").json()[0]
    assert calibrated["source_profile"] == {"scale": .5, "exposure": 2.}
    assert calibrated["ref"] == original["ref"]
    assert calibrated["fit"]["palm"] == pytest.approx(original["fit"]["palm"] * .5, abs=.001)
    assert calibrated["fit"]["tip"] == pytest.approx(original["fit"]["tip"] * .5, abs=.001)
    assert client.get(calibrated["url"]).content == (source / "element.mp4").read_bytes()


def test_invalid_adapter_fails_before_replay_or_library_preview(saved_fire, tmp_path, monkeypatch):
    source, track, spec, receipt = saved_fire
    set_profile(source, receipt, {"exposure": -1.})
    with pytest.raises(ValueError, match="source_profile.exposure"):
        b.make_bending_shot(spec, deny_render, tmp_path / "invalid", track, element_from=source)
    assert not (tmp_path / "invalid" / "composite.mp4").exists()
    monkeypatch.setattr(b, "library_dir", lambda: source.parent)
    with pytest.raises(ValueError, match="source_profile.exposure"):
        TestClient(create_app(None, tmp_path / "jobs")).get("/api/library")


def test_fire_poster_uses_the_visible_peak_instead_of_a_fixed_eight_frame_delay(saved_fire, tmp_path, monkeypatch):
    import cv2
    from cuesync import media

    source, track, spec, _ = saved_fire
    frames = read_video(source / "element.mp4").frames
    frames[4:8, 60:94, 52:77] = [255, 180, 45]
    frames[8:35, 60:94, 52:77] = [75, 25, 5]
    write_video(source / "element.mp4", frames, fps=24)
    encoded = []
    writer = media.write_video

    def observe(path, frames, **kwargs):
        if path.name == "composite.mp4":
            encoded.append(frames.copy())
        return writer(path, frames, **kwargs)

    monkeypatch.setattr(media, "write_video", observe)
    result = b.make_bending_shot(spec, deny_render, tmp_path / "peak", track, element_from=source)
    selected = result.receipt["bending"].get("poster_frame", result.receipt["bending"]["k_launch"] + 8)
    launch = result.receipt["bending"]["k_launch"]
    assert launch <= selected <= launch + 3  # this source peaks briefly, then declines before launch + 8
    assert result.receipt["bending"]["poster_frame"] == selected
    ok, expected = cv2.imencode(".jpg", cv2.cvtColor(encoded[0][selected], cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok and (result.out_dir / "poster.jpg").read_bytes() == expected.tobytes()


@pytest.mark.parametrize("effect,hidden", [("fire_burst", True), ("steam_jet", False)])
def test_a_hidden_fire_or_nonfire_keeps_the_existing_poster_fallback(saved_fire, tmp_path, monkeypatch, effect, hidden):
    source, track, spec, receipt = saved_fire
    receipt["effect"]["id"] = effect
    (source / "receipt.json").write_text(json.dumps(receipt))
    if hidden:
        # A completely occluded source can still illuminate the hand, but is not a visible fire poster candidate.
        monkeypatch.setattr(b, "fire_visibility", lambda *args, box, **kwargs:
                            np.zeros((box[1] - box[0], box[3] - box[2]), np.float32))
    result = b.make_bending_shot(spec, deny_render, tmp_path / "fallback", track, element_from=source)
    assert result.receipt["bending"]["poster_frame"] == result.receipt["bending"]["k_launch"] + 8


def test_equal_visible_fire_peaks_choose_the_first_frame(saved_fire, tmp_path, monkeypatch):
    source, track, spec, _ = saved_fire
    warp, first = b.warp_element, []

    def repeated_frame(*args, **kwargs):
        placed = warp(*args, **kwargs)
        if not first:
            first.append(placed.copy())
        # An exact repeated placed frame avoids codec or volume-flicker differences in this tie fixture.
        return first[0].copy()

    monkeypatch.setattr(b, "warp_element", repeated_frame)
    result = b.make_bending_shot(spec, deny_render, tmp_path / "equal", track, element_from=source)
    assert result.receipt["bending"]["poster_frame"] == result.receipt["bending"]["k_launch"]


@pytest.mark.parametrize("effect", ["fire_burst", "flame_plume"])
def test_charged_fire_keeps_its_ember_envelope_while_selecting_a_poster(saved_fire, tmp_path, make_wav, effect):
    from dataclasses import replace

    source, track, spec, receipt = saved_fire
    receipt["effect"]["id"] = effect
    (source / "receipt.json").write_text(json.dumps(receipt))
    sr = 48000
    audio = np.random.default_rng(5).normal(0., .0005, 2 * sr).astype(np.float32)
    t = np.arange(sr) / sr
    hum = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in (1, 2, 3)) * (.05 + .05 * t)
    audio[int(.25 * sr):int(1.25 * sr)] += hum.astype(np.float32)
    boom_t = np.arange(int(.3 * sr)) / sr
    audio[int(1.25 * sr):int(1.25 * sr) + boom_t.size] += (
        .7 * np.sin(2 * np.pi * 110 * boom_t) * np.exp(-5 * boom_t)).astype(np.float32)
    spec = replace(spec, voice=make_wav(tmp_path, audio, sr), cue="voice", holds=None)
    result = b.make_bending_shot(spec, deny_render, tmp_path / "charge-poster", track, element_from=source)
    saved = result.receipt["bending"]
    assert saved["charge"] and saved["charge"]["release"] - saved["charge"]["from"] > 15
    assert saved["poster_frame"] >= saved["charge"]["from"]
    assert saved["poster_frame"] < 48
    frames = read_video(result.composite).frames.astype(float)
    # More than one charged frame must survive before the release, followed by the actual fire.
    start, release = saved["charge"]["from"], saved["charge"]["release"]
    assert (frames[start + 4:release, 65:105, 45:85].mean((1, 2, 3)) > frames[0, 65:105, 45:85].mean() + 1.).sum() > 5
    assert frames[release:release + 4].max() > 100
    assert saved["held"] is not None if effect == "flame_plume" else saved["held"] is None


@pytest.mark.parametrize("exposure", [1., 2.])
def test_held_release_fades_optical_depth_without_fading_emission_twice(saved_fire, tmp_path, monkeypatch, exposure):
    source, track, spec, receipt = saved_fire
    receipt["effect"]["id"] = "flame_plume"
    set_profile(source, receipt, {"exposure": exposure})
    frames = np.zeros((48, 128, 128, 3), np.uint8)
    frames[4:, 60:94, 52:77] = [180, 90, 18]  # no authored decay: the held playback must fade after release
    write_video(source / "element.mp4", frames, fps=24)
    volumes, placed = {}, {}
    volume, warp = b.volume_element, b.warp_element

    def observe_volume(E, t, *args, **kwargs):
        rgb, transmission, info = volume(E, t, *args, **kwargs)
        volumes[round(t * 24)] = (rgb.copy(), transmission.copy(), info["crop"])
        return rgb, transmission, info

    def observe_warp(E, track, ref, t, *args, **kwargs):
        placed[t] = E.copy()
        return warp(E, track, ref, t, *args, **kwargs)

    monkeypatch.setattr(b, "volume_element", observe_volume)
    monkeypatch.setattr(b, "warp_element", observe_warp)
    result = b.make_bending_shot(spec, deny_render, tmp_path / "release", track, element_from=source)
    saved = result.receipt["bending"]
    assert saved["k_launch"] == 12 and saved["held"]["holds"] == [12]
    # Release starts at 24. Its ten-frame squared fade is 1 at24, .81 at25, .25 at29 and zero at34.
    for frame, weight in ((23, 1.), (24, 1.), (25, .81), (29, .25), (33, .01)):
        rgb, transmission, (x0, y0, x1, y1) = volumes[frame]
        actual = placed[frame][y0:y1, x0:x1]
        expected_rgb = rgb if exposure == 1. else rgb * exposure
        np.testing.assert_array_equal(actual[..., :3], expected_rgb)
        if weight == 1.:
            # Full-strength branches retain the exact previous calculation, not merely a close approximation.
            expected_alpha = 1. - transmission if exposure == 1. else 1. - np.clip(transmission, 0., 1.) ** exposure
            np.testing.assert_array_equal(actual[..., 3], expected_alpha)
        else:
            np.testing.assert_allclose(actual[..., 3], 1. - transmission ** (exposure * weight), atol=1e-7)
    assert .24 < volumes[29][0].sum() / volumes[24][0].sum() < .26  # the RGB fade was already applied once
    assert 34 not in placed  # zero-weight release is absent rather than leaving residual extinction
    assert 0 <= saved["poster_frame"] < 48
