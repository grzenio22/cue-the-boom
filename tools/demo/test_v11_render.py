"""Small media-level checks for the demo renderer's comparison and subtitles."""
import unittest
from pathlib import Path
import wave

import numpy as np
from PIL import Image

import build_demo
from build_demo import caption_overlay, render_beat
from cuesync.media import write_video


class RenderChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path("D:/openHiggsfield_tmp/sol12/demo/render_checks")
        cls.work.mkdir(parents=True, exist_ok=True)
        cls.panels = []
        for k, colour in enumerate(((180, 0, 0), (0, 180, 0), (0, 0, 180))):
            frames = np.full((3, 96, 64, 3), colour, np.uint8)
            frames[1:] = np.minimum(frames[1:].astype(int) + 40, 255).astype(np.uint8)
            path = cls.work / f"{k}.mp4"
            sound = .1 * np.sin(2 * np.pi * (200 + k * 100) * np.arange(6000) / 48000)
            write_video(path, frames, 24, audio=sound.astype(np.float32), sample_rate=48000)
            cls.panels.append(dict(src=str(path), start=0, label=str(k)))
        cls.voice = cls.work / "voice.wav"
        # First 1/8 s is silent; the next 1/8 s has an uncompressed 600 Hz rhythm.
        tone = .2 * np.sin(2 * np.pi * 600 * np.arange(6000) / 48000)
        with wave.open(str(cls.voice), "wb") as f:
            f.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            f.writeframes((np.concatenate([np.zeros(6000), tone]) * 32767).astype("<i2").tobytes())

    def test_comparison_keeps_three_performances_in_step_and_uses_selected_voice(self):
        frames, sound = render_beat(dict(type="compare", panels=self.panels, dur=3 / 24,
                                        headline="Same render", subline="Three voices", audio=1))
        self.assertEqual(len(frames), 3)
        self.assertEqual(sound.size, 6000)
        frequency = np.fft.rfftfreq(sound.size, 1 / 48000)[np.argmax(np.abs(np.fft.rfft(sound)))]
        self.assertLess(abs(frequency - 300), 10)
        for x in (350, 950, 1550):
            self.assertGreater(float(frames[2][500, x].mean()), float(frames[0][500, x].mean()) + 20)

    def test_comparison_mix_plays_all_three_frequencies_without_summing_their_peaks(self):
        try:
            _, sound = render_beat(dict(type="compare", panels=self.panels, dur=3 / 24, audio="mix"))
        except ValueError as e:
            self.fail(f"All-voice comparison is unsupported: {e}")
        frequencies = np.fft.rfftfreq(sound.size, 1 / 48000)
        spectrum = np.abs(np.fft.rfft(sound))
        for wanted in (200, 300, 400):
            self.assertGreater(float(spectrum[np.argmin(abs(frequencies - wanted))]), 50)
        self.assertLess(float(abs(sound).max()), .11)

    def test_comparison_highlight_dims_waiting_panels(self):
        frames, _ = render_beat(dict(type="compare", panels=self.panels, dur=3 / 24, highlight=1))
        self.assertLess(float(frames[0][500, 350].mean()), 35)
        self.assertGreater(float(frames[0][500, 950].mean()), 50)

    def test_turns_plays_complete_replacement_voice_windows_with_panel_and_beat_gains(self):
        panel = dict(self.panels[0], dur=3 / 24, audio_src=str(self.voice), audio_from=.125, audio_gain=2)
        _, sound = render_beat(dict(type="turns", panels=[panel, panel], audio_gain=.5))
        self.assertEqual(sound.size, 12000)
        for start in (0, 6000):
            take = sound[start:start + 6000]
            frequency = np.fft.rfftfreq(take.size, 1 / 48000)[np.argmax(np.abs(np.fft.rfft(take)))]
            self.assertLess(abs(frequency - 600), 10)
            self.assertAlmostEqual(float(abs(take).max()), .2, delta=.002)

    def test_turns_and_comparison_keep_the_same_uncropped_panel_geometry(self):
        panels = [dict(q, dur=3 / 24) for q in self.panels]
        comparison, _ = render_beat(dict(type="compare", panels=panels, dur=3 / 24, highlight=0))
        turns, _ = render_beat(dict(type="turns", panels=panels))
        # Active picture has the same bounds; the top and bottom of a portrait stay visible.
        np.testing.assert_array_equal(turns[0][450:600, 100:600], comparison[0][450:600, 100:600])
        self.assertGreater(int(turns[0][809, 350, 0]), 150)

    def test_clip_source_cue_is_only_visible_in_its_time_window(self):
        frames, _ = render_beat(dict(type="clip", src=self.panels[0]["src"], dur=3 / 24,
                                    cue_text="the author's recorded voice: BOOM", cue_at=1 / 24, cue_dur=1 / 24))
        base, _ = render_beat(dict(type="clip", src=self.panels[0]["src"], dur=3 / 24))
        np.testing.assert_array_equal(frames[0], base[0])
        self.assertFalse(np.array_equal(frames[1], base[1]))
        np.testing.assert_array_equal(frames[2], base[2])

    def test_element_reference_shows_requested_source_moment(self):
        frames, _ = render_beat(dict(type="compare", panels=self.panels, dur=3 / 24,
                                    element_src=self.panels[0]["src"], element_at=2 / 24))
        self.assertGreater(int(frames[0][140, 1785, 0]), 200)

    def test_long_narration_uses_timed_caption_phrases_across_a_cut(self):
        collect = getattr(build_demo, "narration_captions", None)
        self.assertIsNotNone(collect, "Renderer must schedule segmented spoken captions")
        beats = [dict(dur=.1, vo=str(self.voice), vo_at=.05, vo_text="A complete transcript.",
                      vo_captions=[[0, .125, "First phrase."], [.125, .25, "Next phrase."]]), dict(dur=.3)]
        captions = collect(beats)
        self.assertEqual(captions, [(.05, .175, "First phrase."), (.175, .3, "Next phrase.")])

    def test_subtitle_is_only_present_during_its_spoken_line(self):
        base = Image.new("RGB", (1920, 1080), (30, 30, 30))
        before = caption_overlay(base.copy(), 0.4, [(0.5, 1.5, "A quiet puff plays small.")])
        during = caption_overlay(base.copy(), 0.6, [(0.5, 1.5, "A quiet puff plays small.")])
        self.assertTrue(np.array_equal(np.asarray(before), np.asarray(base)))
        self.assertFalse(np.array_equal(np.asarray(during), np.asarray(base)))

    def test_room_tone_fills_a_silent_gap_without_changing_audible_sound(self):
        fill_quiet = getattr(build_demo, "fill_quiet", None)
        self.assertIsNotNone(fill_quiet, "Silent stretches need a measured room-tone bed")
        rate = 48000
        original = np.concatenate([np.full(rate, .1), np.zeros(2 * rate), np.full(rate, .1)]).astype(np.float32)
        room = np.random.default_rng(15).normal(0, 1, rate // 2).astype(np.float32)
        mixed = fill_quiet(original, room)
        np.testing.assert_array_equal(mixed[:rate], original[:rate])
        np.testing.assert_array_equal(mixed[3 * rate:], original[3 * rate:])
        center_rms = float(np.sqrt(np.mean(mixed[round(1.2*rate):round(2.8*rate)] ** 2)))
        self.assertAlmostEqual(20 * np.log10(center_rms), -45, delta=.3)
        self.assertEqual(mixed.size, original.size)


if __name__ == "__main__":
    unittest.main()
