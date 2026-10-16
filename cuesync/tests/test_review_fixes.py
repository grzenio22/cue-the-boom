"""Regression tests for the round-1 review findings (rotation, HDR, room tone, compositing, resume...)."""

import json
import shutil
import subprocess
import wave

import numpy as np
import pytest

from cuesync.align import ElementTiming, plan_alignment
from cuesync.audio import CueSheet, NoOnsetError, analyze
from cuesync.composite import composite
from cuesync.effects import load_effect
from cuesync.experiment import run
from cuesync.guide import paint_guide
from cuesync.media import HDRVideoError, read_audio, read_video, write_video
from cuesync.runner import GraphBindError, bind_fluid_graph

SR = 48000
FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg CLI not installed")


def room_tone(seconds, dbfs, seed=1):
    return np.random.default_rng(seed).normal(0, 10 ** (dbfs / 20), int(seconds * SR)).astype(np.float32)


def tone(seconds, amp, f=700):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


# ---------------------------------------------------------------- media: rotation, HDR, stereo
@needs_ffmpeg
@pytest.mark.parametrize("rot", [90, -90])
def test_phone_rotation_matches_ffmpeg_display(tmp_path, rot):
    f = np.zeros((12, 180, 320, 3), np.uint8); f[:, :60, :80] = (255, 0, 0)        # red block at stored top-left
    write_video(tmp_path / "stored.mp4", f, fps=24)
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-display_rotation", str(rot), "-i", str(tmp_path / "stored.mp4"),
                    "-c", "copy", str(tmp_path / "phone.mp4")], check=True)
    raw = subprocess.run([FFMPEG, "-loglevel", "error", "-i", str(tmp_path / "phone.mp4"), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    shown = np.frombuffer(raw, np.uint8).reshape(320, 180, 3)                        # ffmpeg autorotate = what players show
    ours = read_video(tmp_path / "phone.mp4").frames[0]
    assert ours.shape == (320, 180, 3)
    red = lambda a: np.argwhere((a[..., 0] > 180) & (a[..., 1] < 80))
    assert np.abs(red(ours).mean(0) - red(shown).mean(0)).max() < 3
    small = read_video(tmp_path / "phone.mp4", size=(90, 160)).frames[0]
    assert small.shape == (160, 90, 3)


@needs_ffmpeg
def test_hdr_plate_is_rejected_loudly(tmp_path):
    write_video(tmp_path / "sdr.mp4", np.zeros((6, 64, 64, 3), np.uint8), fps=24)
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", str(tmp_path / "sdr.mp4"), "-c:v", "libx264",
                    "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc",
                    str(tmp_path / "hlg.mp4")], check=True)
    with pytest.raises(HDRVideoError):
        read_video(tmp_path / "hlg.mp4")


def test_stereo_identical_channels_match_mono_loudness(tmp_path, make_wav):
    x = tone(1.0, 0.8)
    mono = read_audio(make_wav(tmp_path, x, SR))
    p = tmp_path / "st.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(np.repeat((x * 32767).astype("<i2"), 2).tobytes())
    st = read_audio(p)
    assert np.abs(st).max() <= 1.0 and abs(np.abs(st).max() - np.abs(mono).max()) < 0.01


# ---------------------------------------------------------------- audio: room tone
@pytest.mark.parametrize("dbfs", [-57, -50, -45])
def test_room_tone_only_raises(dbfs):
    with pytest.raises(NoOnsetError):
        analyze(room_tone(3.0, dbfs), SR)


def test_onset_over_room_tone_is_on_time():
    x = room_tone(3.0, -50); x[SR:SR + 7200] += tone(0.15, 0.1)
    cue = analyze(x, SR)
    assert abs(cue.onsets_s[0] - 1.0) <= 1 / 24 and abs(cue.hero_onset_s - 1.0) <= 1 / 24


def test_sustained_hiss_is_one_onset_and_fades_out():
    rng = np.random.default_rng(3)
    hiss = np.convolve(rng.normal(0, 0.15, int(1.5 * SR)), np.ones(4) / 4, mode="same").astype(np.float32)
    x = room_tone(4.0, -50); x[SR:SR + hiss.size] += hiss
    cue = analyze(x, SR)
    assert len(cue.onsets_s) == 1 and abs(cue.onsets_s[0] - 1.0) <= 1 / 24
    end = cue.voice_end_frame - cue.frame_of(cue.onsets_s[0])
    assert abs(end - 36) <= 3                                                       # hiss lasts 1.5 s = 36 frames


def test_vrrrm_then_kaboom_hero_is_the_boom():
    x = room_tone(4.0, -55)
    x[int(0.5 * SR): int(1.4 * SR)] += tone(0.9, 0.15, f=120)                        # "vrrrm" buzz
    x[int(1.5 * SR): int(1.5 * SR) + 7200] += tone(0.15, 0.8, f=300)                 # "KA-BOOM"
    assert abs(analyze(x, SR).hero_onset_s - 1.5) <= 1 / 24


# ---------------------------------------------------------------- align / composite / guide
def test_no_remap_on_short_spans():
    t = ElementTiming(6, 9, np.zeros(1))
    assert plan_alignment(40, t, plate_peak_frame=60).speed == 1.0


def test_luminous_steam_over_bright_plate_has_no_dark_ring():
    yy, xx = np.mgrid[:128, :128]
    r = np.hypot(yy - 64, xx - 64)
    ball = np.clip(1 - r / 50, 0, 1)                                                 # soft radial steam ball
    el = (ball[..., None] * np.array([235, 235, 235]))[None].repeat(3, 0).astype(np.uint8)
    plate = np.full((3, 128, 128, 3), 245, np.uint8)
    t = ElementTiming(0, 1, np.zeros(1))
    out = composite(plate, el, plan_alignment(0, t), t, dense_smoke=False)[1, 64, 64:120, 0].astype(int)
    assert (np.diff(out) >= -1).all()                                               # monotonic from core to plate


def test_dark_warm_smoke_darkens_a_bright_plate():
    plate = np.full((3, 64, 64, 3), 230, np.uint8)
    el = np.zeros((3, 64, 64, 3), np.uint8); el[:, 16:48, 16:48] = (90, 60, 40)
    t = ElementTiming(0, 1, np.zeros(1))
    out = composite(plate, el, plan_alignment(0, t), t)
    assert out[1, 32, 32].mean() < 230


def test_near_black_codec_noise_leaves_plate_bit_exact():
    plate = np.random.default_rng(0).integers(0, 255, (3, 64, 64, 3), dtype=np.uint8)
    el = np.full((3, 64, 64, 3), 2, np.uint8)                                       # codec noise / haze
    t = ElementTiming(0, 1, np.zeros(1))
    assert np.array_equal(composite(plate, el, plan_alignment(0, t), t), plate)


def test_element_end_fades_instead_of_popping():
    plate = np.full((60, 64, 64, 3), 40, np.uint8)
    el = np.zeros((49, 64, 64, 3), np.uint8); el[:, 8:56, 8:56] = 200                # smoke at full size to the end
    t = ElementTiming(0, 20, np.zeros(1))
    out = composite(plate, el, plan_alignment(0, t), t).astype(int)
    jumps = np.abs(np.diff(out.mean(axis=(1, 2, 3))))
    assert jumps.max() < 15                                                         # no one-frame pop at frame 49


def cue_with(env, hero_frame=24, n=200):
    e = np.zeros(n); e[hero_frame:hero_frame + len(env)] = env
    return CueSheet(24, 0.0, n / 24, [hero_frame / 24], hero_frame / 24, e, e / max(e.max(), 1e-9), int(e.argmax()))


def test_burst_holds_peak_until_second_keyframe():
    p = paint_guide(cue_with(np.ones(10)), load_effect("fireball"), (0.5, 0.7))
    fire_rx = {b["e"]: s["rx"] for b in p.blobs for s in b["shapes"] if s["kind"] == "fire"}
    assert fire_rx[17] == fire_rx[9]


def test_burst_peak_painted_with_wider_stride(tmp_path):
    spec = json.loads(open("effects/fireball.json").read()) | {"keyframe_stride": 16}
    (tmp_path / "f16.json").write_text(json.dumps(spec))
    p = paint_guide(cue_with(np.ones(10)), load_effect(tmp_path / "f16.json"), (0.5, 0.7))
    fire_rx = {b["e"]: s["rx"] for b in p.blobs for s in b["shapes"] if s["kind"] == "fire"}
    assert fire_rx[17] >= fire_rx[0] * 3                                             # peak keyframe painted at full size


# ---------------------------------------------------------------- runner / experiment
def test_prompt_without_trigger_is_rejected():
    g = {"1": {"class_type": "LoadVideo", "inputs": {"file": "x"}},
         "2": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "ainvfxfluid, x"}},
         "3": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
         "4": {"class_type": "SaveVideo", "inputs": {}}}
    with pytest.raises(GraphBindError):
        bind_fluid_graph(g, prompt="a fireball", seed=1)


def test_experiment_resumes_from_existing_elements(tmp_path, make_video, make_wav):
    plate = make_video(tmp_path, n=48, fps=24, w=180, h=320, color_fn=lambda i: (200, 200, 200))
    x = np.zeros(2 * SR, np.float32); x[SR // 2: SR // 2 + 7200] = 0.8 * np.sin(np.arange(7200) * 0.1)
    noise = make_wav(tmp_path, x, SR)
    out, done = tmp_path / "out", tmp_path / "done"
    done.mkdir()
    f = np.zeros((49, 1024, 576, 3), np.uint8); f[4:30, 400:600, 200:380] = (255, 140, 0)
    write_video(done / "element_n1_seed1.mp4", f, fps=24)
    m = run(plates=[plate], noises=[noise], offsets=[0.0], effect="fireball", point=(0.5, 0.5), seeds=[1],
            length=49, elements_dir=done, workflow=None, out=out)                    # nothing to render: reuses
    assert len(m["runs"]) == 1 and m["runs"][0]["element_length"] == 49 and not m["runs"][0]["stale_element"]
