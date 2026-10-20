"""Small media-level checks for the demo renderer's comparison and subtitles."""
from contextlib import nullcontext
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from build_demo import caption_overlay, render_beat
from cuesync.media import write_video


class RenderChecks(unittest.TestCase):
    def test_comparison_keeps_three_performances_in_step_and_uses_selected_voice(self):
        work = Path("D:/openHiggsfield_tmp/sol6/demo/render_checks")
        work.mkdir(parents=True, exist_ok=True)
        with nullcontext(work):
            panels = []
            for k, colour in enumerate(((180, 0, 0), (0, 180, 0), (0, 0, 180))):
                frames = np.full((3, 96, 64, 3), colour, np.uint8)
                frames[1:] = np.minimum(frames[1:].astype(int) + 40, 255).astype(np.uint8)
                path = Path(work) / f"{k}.mp4"
                sound = .1 * np.sin(2 * np.pi * (200 + k * 100) * np.arange(6000) / 48000)
                write_video(path, frames, 24, audio=sound.astype(np.float32), sample_rate=48000)
                panels.append(dict(src=str(path), start=0, label=str(k)))
            frames, sound = render_beat(dict(type="compare", panels=panels, dur=3 / 24,
                                            headline="Same render", subline="Three voices", audio=1))
            self.assertEqual(len(frames), 3)
            self.assertEqual(sound.size, 6000)
            frequency = np.fft.rfftfreq(sound.size, 1 / 48000)[np.argmax(np.abs(np.fft.rfft(sound)))]
            self.assertLess(abs(frequency - 300), 10)
            for x in (350, 950, 1550):
                self.assertGreater(float(frames[2][500, x].mean()), float(frames[0][500, x].mean()) + 20)

    def test_subtitle_is_only_present_during_its_spoken_line(self):
        base = Image.new("RGB", (1920, 1080), (30, 30, 30))
        before = caption_overlay(base.copy(), 0.4, [(0.5, 1.5, "A quiet puff plays small.")])
        during = caption_overlay(base.copy(), 0.6, [(0.5, 1.5, "A quiet puff plays small.")])
        self.assertTrue(np.array_equal(np.asarray(before), np.asarray(base)))
        self.assertFalse(np.array_equal(np.asarray(during), np.asarray(base)))


if __name__ == "__main__":
    unittest.main()
