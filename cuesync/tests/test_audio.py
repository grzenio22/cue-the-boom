import numpy as np
import pytest

from cuesync.audio import CueSheet, NoOnsetError, analyze
from cuesync.media import read_audio

SR = 48000


def burst(at_s, dur_s=0.15, amp=0.8, total=3.0, f=900):
    x = np.zeros(int(total*SR), np.float32); s = int(at_s*SR); n = int(dur_s*SR)
    x[s:s+n] = amp*np.sin(2*np.pi*f*np.arange(n)/SR)*np.exp(-np.arange(n)/(0.04*SR)); return x


def test_single_burst_onset_within_one_frame():
    cue = analyze(burst(1.25), SR, offset_s=0.5)
    assert abs(cue.hero_onset_s - 1.75) <= 1/24
    assert cue.frame_of(cue.hero_onset_s) in (41, 42, 43)


def test_onset_at_sample_zero_detected():
    assert analyze(burst(0.0), SR).onsets_s[0] <= 1/24


def test_hero_is_loudest_of_two():
    x = burst(0.5, amp=0.2) + burst(1.5, amp=0.9)
    assert abs(analyze(x, SR).hero_onset_s - 1.5) <= 1/24


def test_aac_phone_sample_keeps_its_main_attack_and_sustained_end():
    """AAC recompression must not make the startup click replace the real performance at 2.5 s."""
    cue = analyze(read_audio("samples/plates/hand_walk.mp4", SR), SR)
    assert [hit.frame for hit in cue.hits()] == [60]
    assert cue.voice_end_frame == 149


def test_rising_tone_envelope_rises_and_peak_late():
    # Amplitude kept below the -6 dBFS envelope ceiling: the envelope is absolute, so a 0.9 peak
    # would clip to 1.0 and plateau.
    t = np.arange(int(2*SR))/SR
    x = (np.linspace(0.01, 0.4, t.size)*np.sin(2*np.pi*300*t)).astype(np.float32)
    cue = analyze(x, SR)
    assert cue.envelope[10] < cue.envelope[30] < cue.envelope[45] and cue.peak_frame >= 44


def test_ultrasonic_only_noise_is_ignored():
    t = np.arange(int(2*SR))/SR
    x = (0.5*np.sin(2*np.pi*15000*t)).astype(np.float32)
    with pytest.raises(NoOnsetError):
        analyze(x, SR)


def test_silence_raises():
    with pytest.raises(NoOnsetError):
        analyze(np.zeros(SR, np.float32), SR)


def test_loudness_is_absolute_not_normalised():
    quiet, loud = analyze(burst(1.0, amp=0.05), SR), analyze(burst(1.0, amp=0.9), SR)
    assert loud.envelope.max() > quiet.envelope.max() + 0.2
    assert loud.envelope.max() > 0.9 and quiet.envelope.max() < 0.8


def test_envelope_length_and_json_roundtrip():
    cue = analyze(burst(1.0), SR, offset_s=0.5)
    assert len(cue.envelope) == int(np.ceil((0.5 + 3.0) * 24))
    back = CueSheet.from_json(cue.to_json())
    assert back.peak_frame == cue.peak_frame and abs(back.hero_onset_s - cue.hero_onset_s) < 1e-3
