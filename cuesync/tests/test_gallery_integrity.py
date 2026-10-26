"""Shipped gallery media and the order seen by a first-time visitor."""

import hashlib
import json
from pathlib import Path

import pytest


SAMPLES = Path(__file__).resolve().parents[2] / "samples"
CURATED_HAND_ORDER = [
    "hand_fist__boooom__explosion",
    "hand_palm__boom__fire_burst",
    "hand_palm__pfpfboom__again",
    "hand_palm__pfx4__again",
    "hand_palm__sadiquecat__again",
    "hand_palm__unfa__again",
    "hand_palm__huw2k8__again",
    "hand_palm__jedimichael__again",
    "hand_walk__fffff__flame_plume",
    "hand_point__fwoooom__fire_breath",
    "hand_palm__hoooo__air_vortex",
    "hand_open__hoooo__air_vortex",
    "hand_point__zzzzz__lightning",
    "elements_water_air_20261024__dark__lightning_arc",
    "elements_water_air_20261024__dark__lightning_ball",
    "hand_cupped__kaboom__fire_burst",
    "hand_open__fff__flame_plume",
    "hand_fist__boom__fire_burst",
    "hand_punch__boom__fire_burst",
]


def test_every_shipped_index_entry_has_its_folder_and_finished_media():
    missing = []
    for index_path in sorted((SAMPLES / "made_ahead").glob("*/index.json")):
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for entry in index.values():
            directory = index_path.parent / entry["dir"]
            if not directory.is_dir():
                missing.append(str(directory.relative_to(SAMPLES)))
            for filename in ("final.mp4", "receipt.json"):
                if not (directory / filename).is_file():
                    missing.append(str((directory / filename).relative_to(SAMPLES)))
    assert not missing, missing


def test_gallery_leads_with_curated_hands_and_keeps_repeat_performances_together(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from cuesync.server import create_app

    index = json.loads((SAMPLES / "made_ahead/bending/index.json").read_text(encoding="utf-8"))
    names_by_id = {"g" + hashlib.sha1(key.encode()).hexdigest()[:11]: entry["dir"]
                   for key, entry in index.items()}
    client = TestClient(create_app(None, tmp_path / "jobs", samples_dir=SAMPLES,
                                   precomputed=SAMPLES / "made_ahead"))

    response = client.get("/api/gallery")

    assert response.status_code == 200
    gallery = response.json()
    hand_names = [names_by_id[shot["id"]] for shot in gallery if shot["id"] in names_by_id]
    assert hand_names == CURATED_HAND_ORDER
    # Hook uses these first three returned shots, with no additional sorting.
    assert [names_by_id[shot["id"]] for shot in gallery[:3]] == CURATED_HAND_ORDER[:3]
    assert [shot.get("again") for shot in gallery[1:8]] == [0, 1, 2, 3, 4, 5, 6]
