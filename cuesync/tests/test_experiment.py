import json

import numpy as np

from cuesync.audio import analyze
from cuesync.cli import main
from cuesync.compare import side_by_side
from cuesync.experiment import place_audio, run
from cuesync.media import write_video


def test_experiment_reuse_elements(tmp_path, make_video, make_wav):
    plate = make_video(tmp_path, n=96, fps=30, w=180, h=320, color_fn=lambda i: (200, 200, 200))
    x = np.zeros(4*48000, np.float32); x[48000:48000+7200] = 0.8*np.sin(np.arange(7200)*0.1)
    noise = make_wav(tmp_path, x, 48000)
    els = tmp_path/"els"; els.mkdir()
    for i in (1, 2):
        for s in (1, 2):
            f = np.zeros((121, 1024, 576, 3), np.uint8); f[8:60, 400:600, 200:380] = (255, 140, 0)
            write_video(els/f"element_n{i}_seed{s}.mp4", f, fps=24)
    m = run(plates=[plate], noises=[noise, noise], offsets=[0.5, 0.3], effect="fireball", point=(0.5, 0.5),
            seeds=[1, 2], length=121, elements_dir=els, workflow=None, out=tmp_path/"out")
    assert (tmp_path/"out"/"metrics.json").exists() and len(m["runs"]) == 4
    assert all(abs(r["measured_onset_error_frames"]) <= 2 for r in m["runs"])
    assert len(list((tmp_path/"out").glob("compare_*_seed*.mp4"))) == 2


def test_side_by_side_syncs_anchors_and_draws_ribbon():
    a = np.full((30, 128, 72, 3), 40, np.uint8); b = np.full((30, 128, 72, 3), 80, np.uint8)
    env_a = np.linspace(0, 1, 30); env_b = np.linspace(1, 0, 30)
    out = side_by_side(a, b, 10, 4, env_a, env_b, ("pf.", "fwoooOOOM"))
    assert out.shape == (36, 128, 144, 3)                           # b padded by 6 so both anchors land on frame 10
    assert (out[20, -40:, :72] != 40).any() and (out[20, -40:, 72:] != 80).any()   # ribbons drawn
    assert (out[5, :30, :72] != 40).any()                           # label burned in


def test_place_audio_negative_offset_trims():
    s = np.ones(48000, np.float32)
    out = place_audio(s, -0.5, n_frames=24)
    assert out.size == 48000 and out[:24000].all() and not out[24000:].any()


def test_cli_analyze_and_guide(tmp_path, make_wav):
    x = np.zeros(3*48000, np.float32); x[48000:48000+7200] = 0.6*np.sin(np.arange(7200)*0.1)
    wav = make_wav(tmp_path, x, 48000)
    timing = tmp_path/"timing.json"
    timing.write_text(json.dumps({"version": 1, "plate_fps": 24, "sample_rate": 48000, "noise_offset_s": 0.52,
                                  "input_latency_s": 0.02, "onset_s_in_noise": 1.0, "face_offset_s": None,
                                  "emission_point": [0.5, 0.6]}))
    assert main(["analyze", str(wav), "--timing", str(timing), "-o", str(tmp_path/"cue.json")]) == 0
    cue = json.loads((tmp_path/"cue.json").read_text())
    assert abs(cue["offset_s"] - 0.5) < 1e-6 and abs(cue["hero_onset_s"] - 1.5) <= 1/24
    assert main(["guide", "--cue", str(tmp_path/"cue.json"), "--effect", "smoke_puff", "--point", "0.5,0.6",
                 "--length", "49", "-o", str(tmp_path/"g.mp4")]) == 0
    assert (tmp_path/"g.json").exists()
