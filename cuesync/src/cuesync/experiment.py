"""Day-2 experiment: N performances x seeds -> elements -> measured, aligned composites on every plate,
plus contact sheets, per-run metrics and 'same footage, different performance' compares."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .align import plan_alignment
from .audio import analyze
from .compare import side_by_side
from .composite import composite
from .conform import conform
from .effects import load_effect
from .guide import paint_guide, write_guide
from .media import read_audio, read_video, write_video
from .score import measure_composite_onset, measure_element

FPS = 24
SR = 48000


def offset_from_timing(timing: dict) -> float:
    """Dub Mode timing.json -> plate time of noise sample 0 (input latency subtracted)."""
    return float(timing["noise_offset_s"]) - float(timing.get("input_latency_s", 0.0))


def place_audio(samples: np.ndarray, offset_s: float, n_frames: int, fps: int = FPS, sr: int = SR) -> np.ndarray:
    """Lay the performance onto a silent track as long as the plate, starting at plate time ``offset_s``."""
    out = np.zeros(int(round(n_frames / fps * sr)), dtype=np.float32)
    start = int(round(offset_s * sr))
    src = samples
    if start < 0:
        src, start = samples[-start:], 0
    end = min(out.size, start + src.size)
    if end > start:
        out[start:end] = src[: end - start]
    return out


def _contact_sheet(rows: list[tuple[str, np.ndarray, list[int]]], thumb: tuple[int, int] = (180, 320)) -> Image.Image:
    cols = max(len(r[2]) for r in rows)
    sheet = Image.new("RGB", (cols * thumb[0], len(rows) * thumb[1]), (0, 0, 0))
    for ri, (_, frames, picks) in enumerate(rows):
        for ci, f in enumerate(picks):
            f = min(max(f, 0), frames.shape[0] - 1)
            sheet.paste(Image.fromarray(frames[f]).resize(thumb), (ci * thumb[0], ri * thumb[1]))
    return sheet


def run(
    *,
    plates: list,
    noises: list,
    offsets: list[float] | None = None,
    timings: list | None = None,
    effect: str,
    point: tuple[float, float],
    seeds: list[int],
    length: int = 121,
    elements_dir=None,
    workflow=None,
    out,
    labels: list[str] | None = None,
    plate_size: tuple[int, int] = (1080, 1920),
    element_size: tuple[int, int] = (576, 1024),
) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    eff = load_effect(effect)
    if timings:
        offsets = [offset_from_timing(json.loads(Path(t).read_text()) if not isinstance(t, dict) else t) for t in timings]
    if offsets is None:
        offsets = [0.0] * len(noises)
    if len(offsets) != len(noises):
        raise ValueError(f"{len(noises)} noises but {len(offsets)} offsets")
    labels = labels or [Path(n).stem for n in noises]

    plate_frames = {Path(p).stem: conform(read_video(p), size=plate_size, fps=FPS) for p in plates}
    preview = (plate_size[0] // 2, plate_size[1] // 2)
    runs: list[dict] = []
    per_plate_rows: dict[str, list] = {s: [] for s in plate_frames}
    comp_paths: dict[tuple[str, int, int], tuple[Path, int, np.ndarray]] = {}

    for i, (noise, off) in enumerate(zip(noises, offsets), start=1):
        samples = read_audio(noise, SR)
        cue = analyze(samples, SR, offset_s=off, fps=FPS)
        (out / f"cue_n{i}.json").write_text(json.dumps(cue.to_json(), indent=1))
        plan = paint_guide(cue, eff, point, size=element_size, length=length)
        guide_path = out / f"guide_n{i}.mp4"
        write_guide(plan, guide_path, fps=FPS)
        (out / f"guide_n{i}.json").write_text(json.dumps(plan.to_json(), indent=1))
        for seed in seeds:
            name = f"element_n{i}_seed{seed}.mp4"
            render_s = None
            done = out / name
            if done.exists() and done.stat().st_size > 0:  # resume: never re-render (or re-bill) an element
                el_path = done
            elif elements_dir is not None and (Path(elements_dir) / name).exists():
                el_path = Path(elements_dir) / name
            else:
                if workflow is None:
                    raise FileNotFoundError(f"{name} not found in {elements_dir} and no --workflow given to render it")
                from .runner import render_element  # needs comfy_sdk

                res = render_element(guide_path, prompt=eff.prompt, seed=seed, workflow_path=workflow, out_path=done)
                el_path, render_s = res.path, round(res.seconds, 2)
            element = read_video(el_path).frames
            timing = measure_element(element)
            # Peak time-remap only for sustained effects (plume, jet): a burst/puff "peak" is just its last frame.
            sustained = eff.behaviour in ("plume", "jet")
            alignment = plan_alignment(plan.anchor_frame, timing, plate_peak_frame=cue.peak_frame if sustained else None)
            audio = None
            for stem, pf in plate_frames.items():
                comp = composite(pf, element, alignment, timing)
                if audio is None or audio.size != int(round(len(pf) / FPS * SR)):
                    audio = place_audio(samples, off, len(pf))
                comp_path = out / f"comp_{stem}_n{i}_seed{seed}.mp4"
                write_video(comp_path, comp, fps=FPS, audio=audio)
                onset = measure_composite_onset(pf, comp)
                start = alignment.offset + timing.visible_onset
                peak_plate = start + int(round((timing.peak - timing.visible_onset) / alignment.speed))
                runs.append({
                    "plate": stem, "noise": Path(noise).name, "noise_index": i, "seed": seed,
                    "render_s": render_s, "anchor_frame": plan.anchor_frame,
                    "anchor_s": round(plan.anchor_frame / FPS, 4), "audio_onsets_s": [round(t, 4) for t in cue.onsets_s],
                    "visible_onset": timing.visible_onset, "peak": timing.peak,
                    "offset": alignment.offset, "speed": round(alignment.speed, 4),
                    # composite vs anchor: did alignment + matte put the visible effect on the cue frame?
                    "measured_onset_error_frames": None if onset is None else onset - plan.anchor_frame,
                    "peak_error_frames": (peak_plate - cue.peak_frame) if sustained else None,
                })
                small = read_video(comp_path, size=preview).frames
                picks = [plan.anchor_frame - 6, plan.anchor_frame, plan.anchor_frame + 6, plan.anchor_frame + 18, peak_plate]
                per_plate_rows[stem].append((f"n{i}_seed{seed}", small, picks))
                comp_paths[(stem, i, seed)] = (comp_path, plan.anchor_frame, cue.envelope)
                del comp
            del element

    for stem, rows in per_plate_rows.items():
        if rows:
            _contact_sheet(rows).save(out / f"contact_{stem}.png")

    if len(noises) >= 2:
        for stem in plate_frames:
            for seed in seeds:
                pa, aa, ea = comp_paths[(stem, 1, seed)]
                pb, ab, eb = comp_paths[(stem, 2, seed)]
                frames = side_by_side(read_video(pa, size=preview).frames, read_video(pb, size=preview).frames,
                                      aa, ab, ea, eb, (labels[0], labels[1]))
                write_video(out / f"compare_{stem}_seed{seed}.mp4", frames, fps=FPS)

    metrics = {"effect": eff.id, "point": list(point), "length": length, "runs": runs}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics
