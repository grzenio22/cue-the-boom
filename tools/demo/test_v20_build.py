"""Live-take integration regressions: the phone clips and the shots they made replace only the automated Studio."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

import v20_build


def tone_clip(path: Path, seconds: float, bursts: list[float], size: str = "320x180", rotate: int = 0,
              gains: list[float] | None = None) -> None:
    """A test picture with sound: silence with a 0.3 s tone at each of ``bursts`` (s), as loud as its gain."""
    expr = "+".join(f"{g}*between(t,{b},{b + .3})" for b, g in zip(bursts, gains or [1.0] * len(bursts))) or "0"
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
        self.assertEqual([b["id"] for b in data["beats"]], ["live_cold_open", "live_preview", "live_result"])
        self.assertEqual(data["beats"][0]["slot"], "open")
        self.assertLessEqual(data["beats"][0]["dur"], v20_build.OPEN_MAX_S)
        self.assertAlmostEqual(data["beats"][1]["start"], 1.0, delta=.06)  # a second before the first loud sound
        self.assertEqual(data["beats"][1]["headline"], v20_build.LIVE_HEADLINE)

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

    def test_a_loud_take_is_brought_down_to_the_level_of_the_cut(self):
        clip = self.root / "shout.mp4"
        tone_clip(clip, 3, [1.0])  # a 0.5-amplitude tone: loud part about -9 dBFS
        gain = v20_build.level_gain(clip, 0, 3)
        self.assertAlmostEqual(20 * np.log10(gain), v20_build.LIVE_RMS_DB + 9.0, delta=1.0)
        self.assertLessEqual(0.5 * gain, 10 ** (v20_build.LIVE_PEAK_DB / 20))

    def test_live_beats_replace_the_studio_block_in_place(self):
        baseline = {"beats": [{"id": "hero"}, *[{"id": i} for i in v20_build.STUDIO_IDS], {"id": "credits"}]}
        live = {"beats": [{"id": "live_preview"}, {"id": "live_result"}]}
        self.assertEqual(v20_build.expected_ids(baseline, live), ["hero", "live_preview", "live_result", "credits"])

    def test_a_cold_open_plays_first_and_the_rest_replace_the_studio_block(self):
        baseline = {"beats": [{"id": "hero"}, *[{"id": i} for i in v20_build.STUDIO_IDS], {"id": "credits"}]}
        live = {"beats": [{"id": "live_cold_open", "slot": "open"}, {"id": "live_preview"}, {"id": "live_result"}]}
        self.assertEqual(v20_build.expected_ids(baseline, live),
                         ["live_cold_open", "hero", "live_preview", "live_result", "credits"])
        self.assertNotIn("slot", v20_build.live_beat({"id": "live_cold_open", "slot": "open", "src": "x.mp4"},
                                                     {"phones": {}, "jobs": {}}))


    def test_the_cold_open_lands_on_the_boom_not_the_pf_and_reads_with_the_sound_off(self):
        phone = self.root / "IMG_0002.mov"
        tone_clip(phone, 6, [1.0, 2.5], gains=[0.15, 1.0])  # a small pf, then a loud BOOM
        cold = v20_build.ingest([phone], [str(self.job("def456"))])["beats"][0]
        self.assertAlmostEqual(cold["start"], 2.5 - 0.8, delta=.06)
        self.assertEqual(cold["pops"][0][1], "BOOM")
        self.assertAlmostEqual(cold["pops"][0][0], 0.8, delta=.06)
        self.assertTrue(cold["wave"])
        self.assertEqual(cold["headline"], v20_build.OPEN_HEADLINE)

    def test_a_live_beat_can_stand_in_for_one_v19_beat(self):
        baseline = {"beats": [{"id": "hero"}, *[{"id": i} for i in v20_build.STUDIO_IDS], {"id": "air"}, {"id": "credits"}]}
        live = {"beats": [{"id": "live_preview"}, {"id": "live_air", "slot": "replace:air"}]}
        self.assertEqual(v20_build.expected_ids(baseline, live), ["hero", "live_preview", "live_air", "credits"])

    def test_v19_changes_only_by_v20s_recorded_edits(self):
        vo = self.root / "vo20"
        vo.mkdir()
        (vo / "provenance.json").write_text(json.dumps({"lines": {"nodes20": "My nodes.", "contrast20": "Others."}}))
        (vo / "caption_phrases.json").write_text(json.dumps({"nodes20": [[0, 1, "My nodes."]],
                                                             "contrast20": [[0, 1, "Others."]]}))
        self.addCleanup(setattr, v20_build, "VO20", v20_build.VO20)
        v20_build.VO20 = vo
        turns = {"id": "performance_turns", "type": "turns", "dur": 10.0, "panels": [
            {"label": "BOOM", "start": 0.5, "dur": 3, "audio_from": 0.5, "pops": [[2.17, "BOOM"]]},
            {"label": "pf · pf · BOOM", "start": 0, "dur": 3.5}, {"label": "pf pf pf PF", "start": 0, "dur": 3.5}]}
        names = ("comfy_model", "comfy_sampler", "comfy_output", "comfy_cuesync", "dino", "lanterns", "credits")
        baseline = {"version": 19, "duration": 27.5, "beats": [turns, *[{"id": i, "type": "image", "dur": 2.5} for i in names]]}
        beats = v20_build.edited(baseline["beats"])
        self.assertEqual([b["id"] for b in beats], ["performance_turns", "comfy_model", "comfy_sampler", "comfy_nodes",
                                                    "comfy_nodes_report", "dino", "contrast", "lanterns", "credits"])
        boom = beats[0]["panels"][0]
        self.assertEqual((beats[0]["dur"], boom["start"], boom["dur"], boom["audio_from"]), (9.0, 1.5, 2.0, 1.5))
        self.assertAlmostEqual(boom["pops"][0][0], 1.17)
        self.assertEqual(beats[5]["source_gain"], 0.33)
        self.assertEqual(beats[4]["vo_text"], "My nodes.")
        board = {"version": 20, "beats": beats, "v20_edits": v20_build.edits_record()}
        _, errors = v20_build.structure_check(board, baseline)
        self.assertFalse([e for e in errors if "v20's" in e or "v19 beat" in e], errors)
        beats[5]["source_gain"] = 1.0  # a change nobody recorded
        _, errors = v20_build.structure_check(board, baseline)
        self.assertIn("A v19 beat changed beyond v20's recorded edits: dino", errors)
        record = self.root / "live.json"
        record.write_text(json.dumps({"beats": [{"id": "live_a", "slot": "replace:comfy_output"},
                                                {"id": "live_b", "slot": "replace:dino"},
                                                {"id": "live_c", "slot": "replace:dino"}], "phones": {}, "jobs": {}}))
        board["live"] = {"record": str(record)}
        _, errors = v20_build.structure_check(board, baseline)
        self.assertIn("A live beat replaces no v19 beat it may: comfy_output", errors)  # v20 removed it
        self.assertIn("Two live beats replace the same v19 beat", errors)


if __name__ == "__main__":
    unittest.main()
