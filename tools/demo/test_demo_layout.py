"""A scrolling Studio heading must never enter the editorial chip's band."""

import numpy as np
import json
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import build_demo as demo


class ScreenClip:
    def __init__(self, *args, **kwargs):
        self.frames = np.full((1, 720, 1280, 3), (0, 255, 0), np.uint8)

    def frame(self, _):
        return self.frames[0]

    def sound(self):
        return np.zeros(demo.SR // demo.FPS, np.float32)


class HeaderLayoutTest(unittest.TestCase):
    def test_later_integration_cut_preserves_edit_and_audio(self):
        from check_v15 import structure_check

        baseline = json.loads(Path(__file__).with_name("storyboard.json").read_text(encoding="utf-8"))
        board = copy.deepcopy(baseline)
        board["version"] = 17
        next(b for b in board["beats"] if b["id"] == "intro_raw")["stamp"] = (
            "LTX-2.5 fire · CPU temporal derivative · slow motion")
        next(b for b in board["beats"] if b["id"] == "how")["labels"][0] = "LTX fire · CPU derivative"
        _, errors = structure_check(board, baseline)
        self.assertEqual(errors, [])
        board["beats"][0]["dur"] += 1 / demo.FPS
        _, errors = structure_check(board, baseline)
        self.assertTrue(any("timing" in error or "duration" in error for error in errors))

    def test_chip_header_keeps_entire_source_below_chip(self):
        board = json.loads(Path(__file__).with_name("storyboard.json").read_text(encoding="utf-8"))
        for beat in (b for b in board["beats"] if b["id"] in {"upload", "perform", "make"}):
            with self.subTest(chip=beat["chip"]), patch.object(demo, "Clip", ScreenClip):
                configuration = dict(type="clip", src="screen.mp4", dur=1 / demo.FPS,
                                     fill=beat["fill"], chip=beat["chip"], chip_header=beat["chip_header"])
                if beat.get("wave"):
                    configuration.update(wave=True, wave_box=beat["wave_box"])
                    bx, by, bw, bh = beat["wave_box"]
                    self.assertLessEqual((by + bh) * demo.H, 170, "Waveform covers the recorded UI")
                    self.assertGreaterEqual(bx * demo.W, 900, "Waveform covers the step chip")
                frames, _ = demo.render_beat(configuration)
                image = frames[0]
                source = np.all(image == (0, 255, 0), axis=2)
                self.assertFalse(source[:170].any(), "Recorded UI entered the chip's reserved header")
                self.assertTrue(source[170:].any(), "The Studio source disappeared")
                self.assertTrue(np.any(np.all(image[:170] == demo.ACCENT, axis=2)), "The step chip disappeared")
                ys, xs = np.nonzero(source)
                # All four corners of the fitted source remain visible, without a cover crop.
                self.assertTrue(np.all(source[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))


if __name__ == "__main__":
    unittest.main()
