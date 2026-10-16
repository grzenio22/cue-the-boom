"""Effect specs: one JSON file per effect in the repo's effects/ directory (PR-able)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

PROMPT_PREFIX = "ainvfxfluid, "
_ANCHORS = {"hero", "first"}
_BEHAVIOURS = {"burst", "plume", "puff", "jet"}
_FLUIDS = {"fire", "smoke", "steam"}
_COMPOSITES = {"screen", "over"}


@dataclass(frozen=True)
class EffectSpec:
    id: str
    anchor: Literal["hero", "first"]
    behaviour: Literal["burst", "plume", "puff", "jet"]
    fluid: Literal["fire", "smoke", "steam"]
    secondary: Literal["smoke"] | None
    prompt: str
    r_min: float  # fraction of element width
    r_max: float
    rise: float  # fraction of element height risen by the last frame
    keyframe_stride: int  # multiple of 8
    composite: Literal["screen", "over"]

    def to_json(self) -> dict:
        return asdict(self)


def _candidate_dirs(effects_dir: Path) -> list[Path]:
    dirs = [Path(effects_dir)]
    if os.environ.get("CUESYNC_EFFECTS"):
        dirs.append(Path(os.environ["CUESYNC_EFFECTS"]))
    dirs.append(Path(__file__).resolve().parents[3] / "effects")  # repo checkout: <repo>/effects
    return dirs


def _validate(d: dict, source: str) -> EffectSpec:
    required = {f for f in EffectSpec.__dataclass_fields__}
    missing = required - d.keys()
    if missing:
        raise ValueError(f"{source}: missing fields {sorted(missing)}")
    if not str(d["prompt"]).startswith(PROMPT_PREFIX):
        raise ValueError(f"{source}: prompt must start with {PROMPT_PREFIX!r}")
    stride = d["keyframe_stride"]
    if not isinstance(stride, int) or stride <= 0 or stride % 8:
        raise ValueError(f"{source}: keyframe_stride must be a positive multiple of 8, got {stride!r}")
    if not float(d["r_min"]) < float(d["r_max"]):
        raise ValueError(f"{source}: r_min must be < r_max")
    for key, allowed in (("anchor", _ANCHORS), ("behaviour", _BEHAVIOURS), ("fluid", _FLUIDS), ("composite", _COMPOSITES)):
        if d[key] not in allowed:
            raise ValueError(f"{source}: {key} must be one of {sorted(allowed)}, got {d[key]!r}")
    if d["secondary"] not in (None, "smoke"):
        raise ValueError(f"{source}: secondary must be null or 'smoke'")
    return EffectSpec(
        id=str(d["id"]), anchor=d["anchor"], behaviour=d["behaviour"], fluid=d["fluid"],
        secondary=d["secondary"], prompt=str(d["prompt"]), r_min=float(d["r_min"]), r_max=float(d["r_max"]),
        rise=float(d["rise"]), keyframe_stride=int(stride), composite=d["composite"],
    )


def load_effect(id_or_path: str | Path, effects_dir: Path = Path("effects")) -> EffectSpec:
    """Load an effect by id (looked up as <effects_dir>/<id>.json, then fallbacks) or by file path."""
    p = Path(id_or_path)
    if p.suffix == ".json" or p.is_file():
        if not p.is_file():
            raise FileNotFoundError(p)
        return _validate(json.loads(p.read_text(encoding="utf-8")), str(p))
    for d in _candidate_dirs(effects_dir):
        f = d / f"{id_or_path}.json"
        if f.is_file():
            return _validate(json.loads(f.read_text(encoding="utf-8")), str(f))
    raise FileNotFoundError(f"effect {id_or_path!r} not found in {[str(d) for d in _candidate_dirs(effects_dir)]}")
