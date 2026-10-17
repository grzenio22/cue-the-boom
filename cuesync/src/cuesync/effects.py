"""Effect specs: one JSON file per effect in the repo's effects/ directory (PR-able)."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

PROMPT_PREFIX = "ainvfxfluid, "
_ANCHORS = {"hero", "first"}
_BEHAVIOURS = {"burst", "plume", "puff", "jet", "extinguish"}
_FLUIDS = {"fire", "smoke", "steam"}
_COMPOSITES = {"screen", "over"}
_ALIGNS = {"guide", "hit", "onset"}
_GATES = {"both", "attack"}

# Per-behaviour defaults for the optional fields, so older effect files keep working.
_DEFAULTS = {
    #          align    lead  voice_gate  holdout_below  tail_frames  smoke_albedo
    "burst": ("hit", 1, "attack", 0.08, 80, 0.3),  # a fireball's smoke is soot, and it lingers
    "puff": ("hit", 0, "attack", 0.06, 72, 0.85),
    "plume": ("guide", 0, "both", 0.02, 24, 0.4),
    "jet": ("guide", 0, "both", 0.02, 16, 0.95),  # steam
    "extinguish": ("hit", 0, "attack", 0.02, 40, 0.85),  # blowing a flame out (scene mode only)
}


@dataclass(frozen=True)
class SceneSpec:
    """How the effect is painted into the shot itself (scene mode, see scene.py)."""

    prompt: str  # "ainvfxfluid, ..." with {subject} and {towards} filled in per shot
    r_min: float  # size at a whisper / at full voice, fractions of the frame's short side
    r_max: float
    direction: float = 0.0  # default direction, degrees clockwise from straight up (-90 left, 90 right)
    resting: float = 0.0  # a flame that is already there (a lit candle): its height, fraction of the short side
    bend: float = 0.0  # how far a sideways jet curls up along its length (buoyancy), fraction of its length
    shake: float = 0.0  # camera shake after each hit (a blast), peak offset as a fraction of the frame width
    flash: float = 0.0  # exposure flash at each blast's peak (0.4 = +40 % for a frame, half that the next)
    rhythm: str = ""  # put at {rhythm} in the prompt when the take is a rhythm of hits ("pf, pf, KA-BOOM")
    single: str = ""  # put there when it is one hit or one sustained sound
    small: str = ""  # the whole prompt instead, when a single burst is small (a quick quiet "pf")


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
    # How the element is placed: "guide" keeps element frame e on plate frame anchor+e (voice-shaped effects),
    # "hit" puts the element's perceived hit lead_frames before the sound, "onset" puts its first visible frame on it.
    align: Literal["guide", "hit", "onset"] = "guide"
    lead_frames: int = 0
    # "both": the effect also ends with the voice (compositor release); "attack": only its start follows the voice.
    voice_gate: Literal["both", "attack"] = "both"
    release_fire: int = 5  # frames for fire to collapse after the voice stops
    release_smoke: int = 24  # frames for smoke/steam to disperse after the voice stops
    holdout_below: float | None = 0.02  # cut the element this far (fraction of height) below the emission point
    tail_frames: int = 24  # frames rendered after the voice ends (element length is per performance)
    smoke_albedo: float = 0.85  # how much room light the smoke reflects: 0.3 soot .. 0.95 steam
    scene: SceneSpec | None = None  # None: element mode only

    def to_json(self) -> dict:
        return asdict(self)


def _candidate_dirs(effects_dir: Path) -> list[Path]:
    dirs = [Path(effects_dir)]
    if os.environ.get("CUESYNC_EFFECTS"):
        dirs.append(Path(os.environ["CUESYNC_EFFECTS"]))
    dirs.append(Path(__file__).resolve().parents[3] / "effects")  # repo checkout: <repo>/effects
    return dirs


def _validate(d: dict, source: str) -> EffectSpec:
    required = {k for k, f in EffectSpec.__dataclass_fields__.items() if f.default is dataclasses.MISSING}
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
    align, lead, gate, holdout, tail, albedo = _DEFAULTS[d["behaviour"]]
    align, gate = d.get("align", align), d.get("voice_gate", gate)
    if align not in _ALIGNS:
        raise ValueError(f"{source}: align must be one of {sorted(_ALIGNS)}, got {align!r}")
    if gate not in _GATES:
        raise ValueError(f"{source}: voice_gate must be one of {sorted(_GATES)}, got {gate!r}")
    ints = {k: d.get(k, v) for k, v in (("lead_frames", lead), ("release_fire", 5), ("release_smoke", 24), ("tail_frames", tail))}
    for k, v in ints.items():
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"{source}: {k} must be a non-negative integer, got {v!r}")
    scene = None
    if d.get("scene") is not None:
        sd = d["scene"]
        try:
            scene = SceneSpec(prompt=str(sd["prompt"]), r_min=float(sd["r_min"]), r_max=float(sd["r_max"]),
                              direction=float(sd.get("direction", 0.0)), resting=float(sd.get("resting", 0.0)),
                              bend=float(sd.get("bend", 0.0)), shake=float(sd.get("shake", 0.0)),
                              flash=float(sd.get("flash", 0.0)),
                              rhythm=str(sd.get("rhythm", "")), single=str(sd.get("single", "")),
                              small=str(sd.get("small", "")))
        except KeyError as exc:
            raise ValueError(f"{source}: scene needs {exc.args[0]!r}") from exc
        if not scene.prompt.startswith(PROMPT_PREFIX) or (scene.small and not scene.small.startswith(PROMPT_PREFIX)):
            raise ValueError(f"{source}: scene prompts must start with {PROMPT_PREFIX!r}")
        if not 0.0 <= scene.r_min < scene.r_max:
            raise ValueError(f"{source}: scene r_min must be >= 0 and < r_max")
    elif d["behaviour"] == "extinguish":
        raise ValueError(f"{source}: an extinguish effect only exists in scene mode: add a scene section")
    holdout = d.get("holdout_below", holdout)
    albedo = d.get("smoke_albedo", albedo)
    if not isinstance(albedo, (int, float)) or not 0.0 < float(albedo) <= 1.0:
        raise ValueError(f"{source}: smoke_albedo must be in (0, 1], got {albedo!r}")
    if holdout is not None and not 0.0 <= float(holdout) < 1.0:
        raise ValueError(f"{source}: holdout_below must be null or in [0, 1), got {holdout!r}")
    return EffectSpec(
        id=str(d["id"]), anchor=d["anchor"], behaviour=d["behaviour"], fluid=d["fluid"],
        secondary=d["secondary"], prompt=str(d["prompt"]), r_min=float(d["r_min"]), r_max=float(d["r_max"]),
        rise=float(d["rise"]), keyframe_stride=int(stride), composite=d["composite"],
        align=align, voice_gate=gate, holdout_below=None if holdout is None else float(holdout),
        smoke_albedo=float(albedo), scene=scene, **ints,
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
