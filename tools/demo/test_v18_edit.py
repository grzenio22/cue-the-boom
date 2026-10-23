"""Proof-first editorial regressions: protect the measured performance evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from check_v11 import timeline
from check_v15 import sha256, source_provenance_check, structure_check
from v18_edit import PROOF_STAMP, STUDIO_IDS, capture_provenance_check, hero_beat


class ProofFirstEditTest(unittest.TestCase):
    def setUp(self):
        self.baseline = {"version": 17, "duration": 40, "beats": [
            {"id": "cold_open", "type": "clip", "dur": 7, "src": "old-fire.mp4"},
            {"id": "intro_raw", "type": "clip", "dur": 4, "src": "element.mp4",
             "vo": "once.wav", "vo_text": "Generate once. Perform again.", "vo_at": .2},
            {"id": "intro_finished", "type": "clip", "dur": 6, "src": "current-palm.mp4"},
            {"id": "performance_turns", "type": "turns", "dur": 10,
             "panels": [{"src": "boom.mp4", "dur": 3, "audio_src": "boom.wav"},
                        {"src": "two-puffs.mp4", "dur": 3.5, "audio_src": "two-puffs.wav"},
                        {"src": "four-puffs.mp4", "dur": 3.5, "audio_src": "four-puffs.wav"}]},
            {"id": "small_and_big", "type": "compare", "dur": 4},
            {"id": "workflow", "type": "image", "dur": 3, "src": "workflow.png"},
            *[{"id": identity, "type": "clip", "dur": 1, "src": "old-ui.mp4"}
              for identity in ("upload", "tap", "perform", "make", "result")],
            {"id": "credits", "type": "image", "dur": 1, "src": "credits.png"},
        ]}
        proof = copy.deepcopy(self.baseline["beats"][3:5])
        for beat in proof:
            beat["stamp"] = PROOF_STAMP
        studio = [{"id": identity, "type": "clip", "dur": 2, "src": "new-ui.mp4",
                   "chip_header": True, "stamp": "Automated Chromium · virtual microphone · author's recorded voice"}
                  for identity in STUDIO_IDS]
        self.board = {"version": 18, "beats": [hero_beat(self.baseline), *proof, *studio,
                                              copy.deepcopy(self.baseline["beats"][5]),
                                              copy.deepcopy(self.baseline["beats"][-1])]}
        self.board["duration"] = timeline(self.board["beats"])[-1]["end"]

    def test_complete_proof_and_fresh_flow_come_first(self):
        report, errors = structure_check(self.board, self.baseline)
        self.assertEqual(errors, [])
        self.assertEqual(report["proof_complete_seconds"], 14.75)
        self.assertEqual(report["studio_start_seconds"], 18.75)
        self.assertIn("slow motion", self.board["beats"][0]["stamp"])

    def test_old_explosion_or_silent_narration_cannot_return(self):
        for field, value in (("src", "old-fire.mp4"), ("vo", None), ("stamp", "Live"), ("start", 0)):
            changed = copy.deepcopy(self.board)
            changed["beats"][0][field] = value
            self.assertTrue(structure_check(changed, self.baseline)[1])

    def test_proof_audio_rhythm_duration_and_source_remain_exact(self):
        for field, value in (("audio_src", "new-voice.wav"), ("dur", 1), ("src", "different-output.mp4")):
            changed = copy.deepcopy(self.board)
            changed["beats"][1]["panels"][0][field] = value
            self.assertTrue(structure_check(changed, self.baseline)[1])

    def test_automated_capture_disclosure_and_real_action_speed_are_required(self):
        for field, value in (("stamp", "Live microphone"), ("speed", 4), ("chip_header", False)):
            changed = copy.deepcopy(self.board)
            changed["beats"][3][field] = value
            self.assertTrue(structure_check(changed, self.baseline)[1])

    def test_workflow_cannot_displace_opening_proof(self):
        changed = copy.deepcopy(self.board)
        changed["beats"].insert(0, changed["beats"].pop(-2))
        self.assertTrue(structure_check(changed, self.baseline)[1])


class PreservedSourcesTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.code = self.root / "server.py"
        self.code.write_bytes(b"previous implementation")
        code = {"server.py": sha256(self.code)}
        self.input = self.root / "input.mp4"
        self.input.write_bytes(b"source input bytes")
        self.output = self.root / "clips" / "composite.mp4"
        self.output.parent.mkdir()
        self.output.write_bytes(b"approved composite bytes")
        inputs = {str(self.input): sha256(self.input)}
        job = {"inputs_sha256": inputs, "outputs_sha256": {str(self.output): sha256(self.output)},
               "fingerprint": hashlib.sha256(json.dumps({"code": code, "inputs": inputs}, sort_keys=True).encode()).hexdigest()}
        self.manifest = self.output.parent / "manifest.json"
        self.manifest.write_text(json.dumps({"source_sha256": code, "jobs": {"fixture": job}}))
        self.board = {"version": 18, "provenance": str(self.manifest), "beats": [{"src": str(self.output)}],
                      "source_reuse": {"manifest": str(self.manifest), "manifest_sha256": sha256(self.manifest),
                                       "new_model_generations": 0, "composites_rebuilt_for_this_edit": False}}
        self.code.write_bytes(b"new unrelated endpoint implementation")

    def check(self, board=None):
        with patch("check_v15.CHECKOUT", self.root), patch("check_v15.proof_provenance_check", return_value=([], [])):
            return source_provenance_check(board or self.board)

    def test_immutable_manifest_can_retain_prior_code_fingerprints(self):
        report, errors = self.check()
        self.assertEqual(errors, [])
        self.assertIn("Editorial reuse", report["manifests"][0]["policy"])

    def test_missing_wrong_manifest_hash_or_rebuild_claim_is_rejected(self):
        for field, value in (("manifest_sha256", None), ("manifest_sha256", "0" * 64),
                             ("composites_rebuilt_for_this_edit", True)):
            changed = copy.deepcopy(self.board)
            changed["source_reuse"][field] = value
            errors = self.check(changed)[1]
            self.assertTrue(any("immutable prior source manifest" in error for error in errors), errors)

    def test_prior_input_and_output_bytes_are_still_verified(self):
        for path in (self.input, self.output):
            original = path.read_bytes()
            path.write_bytes(b"changed bytes")
            self.assertTrue(any("differs from manifest" in error for error in self.check()[1]))
            path.write_bytes(original)


class ActualBrowserCaptureTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        assets = {}
        for name in ("raw_screen_video", "recorded_microphone", "actual_final", "downloaded_final",
                     "receipt", "browser_evidence"):
            path = self.root / name
            path.write_bytes(b"actual result" if name in {"actual_final", "downloaded_final"} else name.encode())
            assets[name] = {"path": str(path), "sha256": sha256(path)}
        self.capture = {"passed": True, "assets": assets, "job_id": "new-browser-job",
                        "new_model_generation": False, "actual_cpu_composition_seconds": 18.4,
                        "request": {"fields": {"performance_mode": "reuse"}},
                        "pcm_first_sample_page_relative_s": 5.65,
                        "recorded_voice": {"duration": 3.57}}
        self.manifest = self.root / "capture.json"
        self.board = {"beats": [{"id": identity,
                                  "src": assets["actual_final" if identity == "result" else "raw_screen_video"]["path"]}
                                 for identity in STUDIO_IDS]}
        by_id = {beat["id"]: beat for beat in self.board["beats"]}
        by_id["perform"].update(start=4.3, dur=5.083333, audio_at=1.35, audio_from=0,
                                audio_src=assets["recorded_microphone"]["path"])
        by_id["make"]["chip"] = "18.4s CPU composition · wait shortened"
        self.bind()

    def bind(self):
        self.manifest.write_text(json.dumps(self.capture))
        self.board["browser_capture"] = {"manifest": str(self.manifest), "manifest_sha256": sha256(self.manifest)}

    def test_actual_pcm_download_and_reuse_are_bound(self):
        report, errors = capture_provenance_check(self.board)
        self.assertEqual(errors, [])
        self.assertEqual(report["job_id"], "new-browser-job")

    def test_new_generation_or_different_download_is_rejected(self):
        self.capture["new_model_generation"] = True
        self.bind()
        self.assertTrue(capture_provenance_check(self.board)[1])
        self.capture["new_model_generation"] = False
        self.capture["assets"]["downloaded_final"]["sha256"] = "0" * 64
        self.bind()
        self.assertTrue(capture_provenance_check(self.board)[1])

    def test_truncating_or_shifting_microphone_audio_is_rejected(self):
        for field, value in (("audio_at", 0), ("audio_from", .5), ("dur", 3)):
            board = copy.deepcopy(self.board)
            next(beat for beat in board["beats"] if beat["id"] == "perform")[field] = value
            self.assertTrue(capture_provenance_check(board)[1])


if __name__ == "__main__":
    unittest.main()
