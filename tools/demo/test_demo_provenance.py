"""Verify demo element ancestry from small byte fixtures, without media rendering."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from check_v15 import proof_provenance_check, sha256, source_provenance_check, structure_check


class DemoLineageTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.original = self.write("original/element.mp4", b"original generated pixels")
        self.playback = self.write("playback/element.mp4", b"CPU adapted pixels")
        self.derivative = self.write("compact/element.mp4", self.playback.read_bytes())
        self.adapter = self.write("source-adapter.json", b'{"kind":"playback adapter"}')
        self.recipe = self.write_json("compact/derivative-recipe.json", {
            "original_element_sha256": sha256(self.original),
            "derivative_sha256": sha256(self.playback),
            "new_model_renders": 0, "gpu_used": False,
        })
        files = {"samples/original/element.mp4": self.original,
                 "samples/compact/element.mp4": self.derivative,
                 "samples/compact/derivative-recipe.json": self.recipe}
        self.sidecar = self.write_json("source-provenance.json", {
            "element_sha256": sha256(self.playback),
            "playback_adapter_sha256": sha256(self.adapter),
            "no_new_model_generation_this_rebuild": True, "no_gpu_this_rebuild": True,
            "source_files": {key: sha256(path) for key, path in files.items()},
        })
        self.job = {
            "playback_element": str(self.playback), "element_sha256": sha256(self.playback),
            "no_new_render": True, "no_gpu": True,
            "inputs_sha256": {str(p): sha256(p) for p in
                              [self.playback, self.adapter, self.sidecar, *files.values()]},
            "source_lineage": {
                "provenance_path": str(self.sidecar), "source_adapter_path": str(self.adapter),
                "source_files": {key: {"path": str(path), "sha256": sha256(path)}
                                 for key, path in files.items()},
                "original_generation_element": str(self.original),
                "original_generation_sha256": sha256(self.original),
                "derivative_recipe": str(self.recipe),
            },
        }

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def write_json(self, name, content):
        return self.write(name, json.dumps(content).encode())

    def check(self, job=None, version=17):
        jobs = {name: copy.deepcopy(job or self.job) for name in
                ("proof_boom", "proof_pf_pf_boom", "proof_pf_x4")}
        return proof_provenance_check(jobs, version)

    def test_shared_derivative_with_multiple_element_inputs_passes(self):
        rows, errors = self.check()
        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 3)

    def test_changed_original_and_playback_bytes_are_rejected(self):
        for path in (self.original, self.playback, self.recipe, self.adapter, self.sidecar):
            with self.subTest(path=path):
                saved = path.read_bytes()
                path.write_bytes(b"tampered bytes")
                _, errors = self.check()
                self.assertTrue(any("bytes differ" in e for e in errors), errors)
                path.write_bytes(saved)

    def test_recipe_with_rehashed_but_wrong_ancestry_is_rejected(self):
        recipe = json.loads(self.recipe.read_text())
        recipe["original_element_sha256"] = "0" * 64
        self.recipe.write_text(json.dumps(recipe))
        changed_hash = sha256(self.recipe)
        relative = "samples/compact/derivative-recipe.json"
        self.job["source_lineage"]["source_files"][relative]["sha256"] = changed_hash
        self.job["inputs_sha256"][str(self.recipe)] = changed_hash
        sidecar = json.loads(self.sidecar.read_text())
        sidecar["source_files"][relative] = changed_hash
        self.sidecar.write_text(json.dumps(sidecar))
        self.job["inputs_sha256"][str(self.sidecar)] = sha256(self.sidecar)
        _, errors = self.check()
        self.assertTrue(any("recipe ancestry" in e for e in errors), errors)

    def test_playback_hash_cannot_point_at_a_different_bound_element(self):
        self.job["playback_element"] = str(self.original)
        _, errors = self.check()
        self.assertTrue(any("playback element" in e for e in errors), errors)

    def test_omitted_source_file_and_unbound_adapter_are_rejected(self):
        missing = copy.deepcopy(self.job)
        missing["source_lineage"]["source_files"].pop("samples/original/element.mp4")
        _, errors = self.check(missing)
        self.assertTrue(any("cover exactly" in e for e in errors), errors)
        missing = copy.deepcopy(self.job)
        missing["inputs_sha256"].pop(str(self.adapter))
        _, errors = self.check(missing)
        self.assertTrue(any("not bound" in e and "adapter" in e for e in errors), errors)

    def test_different_original_generation_across_proofs_is_rejected(self):
        jobs = {name: copy.deepcopy(self.job) for name in
                ("proof_boom", "proof_pf_pf_boom", "proof_pf_x4")}
        jobs["proof_pf_x4"]["source_lineage"]["original_generation_sha256"] = "0" * 64
        _, errors = proof_provenance_check(jobs, 17)
        self.assertTrue(any("identical original generation" in e for e in errors), errors)

    def test_v16_legacy_single_element_needs_no_lineage(self):
        legacy = {"element_sha256": sha256(self.playback), "no_new_render": True, "no_gpu": True,
                  "inputs_sha256": {str(self.playback): sha256(self.playback)}}
        _, errors = self.check(legacy, version=16)
        self.assertEqual(errors, [])


class IntegrationDisclosureTest(unittest.TestCase):
    def setUp(self):
        self.baseline = {"version": 16, "duration": 6, "beats": [
            {"id": "intro_raw", "type": "clip", "dur": 1, "stamp": "Raw LTX fire",
             "start": 2.4, "speed": .225},
            {"id": "how", "type": "triptych", "dur": 1,
             "labels": ["LTX over black", "Carried", "Matte", "Light"]},
            {"id": "upload", "type": "clip", "dur": 1, "chip_header": True},
            {"id": "perform", "type": "clip", "dur": 1, "chip_header": True,
             "wave_box": [.54, .025, .4, .115], "audio_gain": 4},
            {"id": "make", "type": "clip", "dur": 1, "chip_header": True},
            {"id": "workflow_fire", "type": "clip", "dur": 1, "start": 2.4, "speed": .45},
        ]}
        self.board = copy.deepcopy(self.baseline)
        self.board["version"] = 17
        self.board["beats"][0]["stamp"] = "LTX-2.5 fire · CPU temporal derivative · slow motion"
        self.board["beats"][0].update(start=2.65, speed=.2)
        self.board["beats"][1]["labels"][0] = "LTX fire · CPU derivative"
        self.board["beats"][5].update(start=2.65, speed=.4)

    def test_exact_derivative_disclosures_are_allowed(self):
        _, errors = structure_check(self.board, self.baseline)
        self.assertEqual(errors, [])

    def test_other_labels_audio_and_timing_are_still_protected(self):
        for index, field, value in ((1, "labels", ["LTX fire · CPU derivative", "Wrong", "Matte", "Light"]),
                                    (3, "audio_gain", 5), (0, "dur", 2)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.board)
                changed["beats"][index][field] = value
                _, errors = structure_check(changed, self.baseline)
                self.assertTrue(errors)

    def test_raw_claim_and_v16_disclosure_change_are_rejected(self):
        changed = copy.deepcopy(self.board)
        changed["beats"][0]["stamp"] = "Raw LTX fire"
        _, errors = structure_check(changed, self.baseline)
        self.assertTrue(any("disclose" in error for error in errors), errors)
        changed = copy.deepcopy(self.board)
        changed["version"] = 16
        _, errors = structure_check(changed, self.baseline)
        self.assertTrue(any("editorial" in error for error in errors), errors)


    def test_only_exact_approved_source_windows_are_allowed(self):
        for index in (0, 5):
            for field, value in (("start", 2.4), ("speed", .3), ("dur", 2),
                                 ("audio_gain", .5), ("vo_text", "Changed narration"),
                                 ("caption", "Changed content")):
                with self.subTest(beat=index, field=field):
                    changed = copy.deepcopy(self.board)
                    changed["beats"][index][field] = value
                    _, errors = structure_check(changed, self.baseline)
                    self.assertTrue(errors)
        changed = copy.deepcopy(self.board)
        changed["beats"][1]["start"] = 2.65
        _, errors = structure_check(changed, self.baseline)
        self.assertTrue(any("editorial" in error for error in errors), errors)


class RecapWindowTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.run = self.root / "run"
        (self.run / "clips").mkdir(parents=True)
        (self.run / "cards").mkdir()
        source = self.root / "renderer.py"
        source.write_text("# hash fixture")
        self.card = self.run / "cards/pipeline_motion.mp4"
        self.card.write_bytes(b"card fixture; no media decoding")
        self.manifest = self.run / "clips/manifest.json"
        self.manifest.write_text(json.dumps({"source_sha256": {"renderer.py": sha256(source)}, "jobs": {}}))

    def check(self, window, version=17):
        provenance = {"outputs_sha256": {str(self.card): sha256(self.card)}}
        if window is not None:
            provenance["source_window"] = window
        (self.run / "cards/provenance.json").write_text(json.dumps(provenance))
        board = {"version": version, "provenance": str(self.manifest), "beats": [{"src": str(self.card)}]}
        with patch("check_v15.CHECKOUT", self.root), patch("check_v15.proof_provenance_check", return_value=([], [])):
            return source_provenance_check(board)

    def test_v17_recap_requires_exact_source_window(self):
        _, errors = self.check({"start_seconds": 2.65, "span_seconds": .8})
        self.assertEqual(errors, [])
        for window in (None, {"start_seconds": 2.4, "span_seconds": .8},
                       {"start_seconds": 2.65, "span_seconds": .9}):
            with self.subTest(window=window):
                _, errors = self.check(window)
                self.assertTrue(any("recap source window" in error for error in errors), errors)

    def test_v16_recap_without_window_remains_valid(self):
        _, errors = self.check(None, version=16)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
