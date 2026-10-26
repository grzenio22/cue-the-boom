"""Live-take integration regressions: the phone clips and the shots they made replace only the automated Studio."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

import v20_build


def tone_clip(path: Path, seconds: float, bursts: list[float], size: str = "320x180", rotate: int = 0) -> None:
    """A test picture with sound: silence with a loud 0.3 s tone at each of ``bursts`` (s)."""
    expr = "+".join(f"between(t,{b},{b + .3})" for b in bursts) or "0"
    raw = path.with_suffix(".raw.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=30:d={seconds}",
                    "-f", "lavfi", "-i", f"aevalsrc=exprs='0.5*sin(2*PI*440*t)*({expr})':s=48000:d={seconds}",
                    "-shortest", "-c:v", "libx264", "-c:a", "aac", str(raw)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", *(["-display_rotation", str(rotate)] if rotate else []), "-i", str(raw),
                    "-c", "copy", "-metadata", "creation_time=2026-10-28T10:00:00Z", str(path)], check=True)
    raw.unlink()


class LiveTakeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        v20_build.run = self.root / "run"
        v20_build.SCRATCH = self.root / "scratch"

    def tearDown(self):
        self.tmp.cleanup()

    def job(self, name: str, size: str = "320x244") -> Path:
        d = self.root / "jobs" / name
        (d / "in").mkdir(parents=True)
        tone_clip(d / "final.mp4", 3.5, [1.0], size=size)
        (d / "receipt.json").write_text(json.dumps({"mode": "bending", "effect": {"id": "fire_burst"},
                                                   "inputs": {"cue": "clip"}, "render": {"seconds": 0.0},
                                                   "bending": {"hits": [{"frame": 24}]}, "seconds_total": 57}))
        (d / "in" / "plate.mp4").write_bytes(b"take")
        return d

    def test_onsets_find_each_loud_sound(self):
        clip = self.root / "boom.mp4"
        tone_clip(clip, 6, [1.0, 3.5])
        found = [o["at"] for o in v20_build.onsets(clip)]
        self.assertEqual(len(found), 2)
        self.assertAlmostEqual(found[0], 1.0, delta=.06)
        self.assertAlmostEqual(found[1], 3.5, delta=.06)

    def test_ingest_turns_a_rotated_phone_clip_upright_and_drafts_a_cut(self):
        phone = self.root / "IMG_0001.mov"
        tone_clip(phone, 5, [2.0], size="180x320", rotate=90)  # stored portrait, shown landscape, as a phone does
        data = v20_build.ingest([phone], [str(self.job("abc123"))])
        record = data["phones"]["phone_1.mp4"]
        self.assertEqual(record["rotation"], 90)
        self.assertEqual(record["upright"], [320, 180])
        self.assertEqual(v20_build.probe(v20_build.live_dir() / "phone_1.mp4")["upright"], [320, 180])
        self.assertTrue((v20_build.live_dir() / "phone_1_sheet.jpg").is_file())
        self.assertEqual(data["jobs"]["abc123"]["first_hit_s"], 1.0)
        self.assertTrue((v20_build.live_dir() / "jobs/abc123/in/plate.mp4").is_file())
        self.assertTrue(data["draft"])
        self.assertEqual([b["id"] for b in data["beats"]], ["live_preview", "live_result"])
        self.assertAlmostEqual(data["beats"][0]["start"], 1.0, delta=.06)  # a second before the first loud sound

    def test_a_shot_loses_its_export_footer_and_a_portrait_one_keeps_its_backdrop(self):
        v20_build.live_dir().mkdir(parents=True)
        landscape = self.job("wide", size="1920x1144")  # a webcam take's shot: 1080 rows and the footer
        tall = self.job("tall", size="1080x1984")       # a phone take's: 1920 rows and the footer
        live = {"phones": {}, "jobs": {"wide": {}, "tall": {}}}
        for name, source in (("wide", landscape), ("tall", tall)):
            dst = v20_build.live_dir() / "jobs" / name
            dst.mkdir(parents=True)
            (dst / "final.mp4").write_bytes((source / "final.mp4").read_bytes())
        wide = v20_build.live_beat({"id": "live_result", "src": "live/jobs/wide/final.mp4", "dur": 3}, live)
        portrait = v20_build.live_beat({"id": "live_result", "src": "live/jobs/tall/final.mp4", "dur": 3}, live)
        self.assertEqual(wide["crop"], [0, 0, 1, round(1080 / 1144, 6)])
        self.assertNotIn("portrait_backdrop", wide)
        self.assertEqual(portrait["crop"], [0, 0, 1, round(1920 / 1984, 6)])
        self.assertTrue(portrait["portrait_backdrop"])

    def test_live_beats_replace_the_studio_block_in_place(self):
        baseline = {"beats": [{"id": "hero"}, *[{"id": i} for i in v20_build.STUDIO_IDS], {"id": "credits"}]}
        live = {"beats": [{"id": "live_preview"}, {"id": "live_result"}]}
        self.assertEqual(v20_build.expected_ids(baseline, live), ["hero", "live_preview", "live_result", "credits"])


if __name__ == "__main__":
    unittest.main()
