"""Decode and check the finished contest cut, then make timecoded contact sheets.

    python tools/demo/check_v11.py

The stream is inspected frame by frame; only the contact-sheet samples are
kept. QA results distinguish intentionally static cards and the cold open's
clean hold from freezes that need a look. The audio checks measure levels and
the presence of the narration waveform; listening and judging the edit remain
human review tasks. All output goes beside the video, not into the inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import av
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cuesync" / "src"))
from cuesync.media import read_audio  # noqa: E402
from build_demo import BG, DUCK, FG, FPS, H, SR, W, caption_overlay, font  # noqa: E402


RUN = Path("D:/openHiggsfield/runs/demo_v11")


def db(value: float) -> float:
    return round(20 * math.log10(max(float(value), 1e-12)), 3)


def levels(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    if not a.size:
        return dict(samples=0, peak=0, peak_dbfs=-240, rms=0, rms_dbfs=-240,
                    near_full_scale_samples=0, near_full_scale_fraction=0)
    peak, rms = float(np.max(np.abs(a))), float(np.sqrt(np.mean(a * a)))
    count = int(np.count_nonzero(np.abs(a) >= .999))
    return dict(samples=int(a.size), peak=round(peak, 7), peak_dbfs=db(peak),
                rms=round(rms, 7), rms_dbfs=db(rms), near_full_scale_samples=count,
                near_full_scale_fraction=round(count / a.size, 8))


def timeline(beats: list[dict]) -> list[dict]:
    start = 0
    result = []
    for i, beat in enumerate(beats):
        n = round(float(beat["dur"]) * FPS)
        result.append(dict(beat=beat, id=beat.get("id", f"beat_{i+1}"),
                           start=start / FPS, end=(start + n) / FPS, frames=n))
        start += n
    return result


def beat_at(schedule: list[dict], seconds: float) -> dict:
    return next((b for b in schedule if b["start"] <= seconds < b["end"]), schedule[-1])


def timecode(seconds: float) -> str:
    n = round(seconds * FPS)
    minutes, remainder = divmod(n, 60 * FPS)
    secs, frame = divmod(remainder, FPS)
    return f"{minutes:02d}:{secs:02d}:{frame:02d}"


def sheet(samples: list[tuple[float, str, Image.Image]], path: Path, columns: int, width: int) -> None:
    thumb_h, label_h, gap = round(width * H / W), 34 if width <= 400 else 42, 8
    rows = math.ceil(len(samples) / columns)
    image = Image.new("RGB", (columns * (width + gap) + gap,
                              rows * (thumb_h + label_h + gap) + gap), BG)
    d = ImageDraw.Draw(image)
    f = font(20 if width <= 400 else 27, True)
    for i, (t, name, source) in enumerate(samples):
        x = gap + (i % columns) * (width + gap)
        y = gap + (i // columns) * (thumb_h + label_h + gap)
        image.paste(source.resize((width, thumb_h), Image.Resampling.LANCZOS), (x, y))
        label = f"{timecode(t)}  {name}"
        while d.textlength(label, font=f) > width - 10 and name:
            name = name[:-1]
            label = f"{timecode(t)}  {name}"
        d.text((x + 5, y + thumb_h + 4), label, font=f, fill=FG)
    image.save(path, quality=92, subsampling=0)


def pronunciation_key(value: str) -> str:
    value = value.lower().replace("two point five", "2.5")
    return re.sub(r"[^a-z0-9]", "", value)


def narration_check(schedule: list[dict], total: float, output: Path) -> tuple[list[dict], list[str]]:
    issues, result, previous_end = [], [], 0.0
    lines_path = output / "vo_lines.json"
    written = json.loads(lines_path.read_text(encoding="utf-8")) if lines_path.exists() else {}
    durations_path = output / "vo11/durations.json"
    declared = json.loads(durations_path.read_text(encoding="utf-8")) if durations_path.exists() else {}
    blank = Image.new("RGB", (W, H), BG)
    for entry in schedule:
        b = entry["beat"]
        if not b.get("vo"):
            continue
        path = Path(b["vo"])
        if not path.is_file():
            issues.append(f"Missing narration: {path}")
            continue
        audio = read_audio(path, SR) * float(b.get("vo_gain", 1))
        begin = entry["start"] + float(b.get("vo_at", .15))
        end = begin + audio.size / SR
        text = b.get("vo_text", "")
        row = dict(id=entry["id"], source=str(path), caption=text,
                   start=round(begin, 6), end=round(end, 6),
                   duration=round(audio.size / SR, 6), source_levels=levels(audio),
                   ends_after_own_beat=round(max(0, end - entry["end"]), 6))
        active = np.flatnonzero(np.abs(audio) > max(.001, (float(np.max(np.abs(audio))) if audio.size else 0) * .01))
        if not active.size:
            issues.append(f"Narration is silent: {entry['id']}")
        row["last_active_sample_before_file_end_seconds"] = round((audio.size - active[-1] - 1) / SR, 6) if active.size else None
        row["source_last_50ms_levels"] = levels(audio[-round(.05 * SR):])
        if not text:
            issues.append(f"Narration lacks a caption: {entry['id']}")
        else:
            if path.stem in written:
                row["caption_matches_spoken_text"] = pronunciation_key(text) == pronunciation_key(written[path.stem])
                if not row["caption_matches_spoken_text"]:
                    issues.append(f"Caption differs from narration input: {entry['id']}")
            try:
                caption_overlay(blank.copy(), .5, [(0, 1, text)])
                row["caption_fits_two_line_safe_area"] = True
            except ValueError as exc:
                row["caption_fits_two_line_safe_area"] = False
                issues.append(str(exc))
        if path.stem in declared:
            row["duration_manifest_error_seconds"] = round(audio.size / SR - declared[path.stem], 6)
            if abs(row["duration_manifest_error_seconds"]) > .002:
                issues.append(f"Narration duration manifest differs: {path.stem}")
        if begin < previous_end - .002:
            issues.append(f"Overlapping narration at {begin:.3f}s: {entry['id']}")
        if end > total + .002:
            issues.append(f"Narration cut at picture end: {entry['id']} ({end-total:.3f}s)")
        previous_end = max(previous_end, end)
        result.append(row)
    return result, issues


def ranges(mask: list[bool], minimum: int = 1) -> list[tuple[int, int]]:
    result, begin = [], None
    for i, active in enumerate(mask + [False]):
        if active and begin is None:
            begin = i
        elif not active and begin is not None:
            if i - begin >= minimum:
                result.append((begin, i))
            begin = None
    return result


def planned_hold(entry: dict, start: float, end: float) -> str | None:
    b = entry["beat"]
    if b["type"] in {"title", "credits", "image"}:
        return "Designed title or still card"
    if entry["id"] == "cold_open" and start >= entry["start"] + 5.20:
        return "Designed clean final frame, after the explosion has cleared"
    return None


def event(begin: int, end: int, schedule: list[dict], kind: str) -> dict:
    start, stop = begin / FPS, end / FPS
    entry = beat_at(schedule, (start + stop) / 2)
    row = dict(kind=kind, start=round(start, 6), end=round(stop, 6),
               duration=round(stop - start, 6), beat=entry["id"])
    reason = planned_hold(entry, start, stop)
    if kind == "freeze" and reason:
        row["classification"] = "planned"
        row["reason"] = reason
    elif kind == "center_black" and (entry["id"] in {"one_generation", "how"}
                                       or entry["beat"]["type"] in {"title", "image", "credits"}):
        row["classification"] = "designed_background"
        row["reason"] = "A raw render over black or the designed card background; inspect the contact sheet"
    else:
        row["classification"] = "review"
    return row


def scan_video(video: Path, schedule: list[dict], narration: list[dict], out: Path) -> dict:
    total = schedule[-1]["end"]
    targets: dict[int, list[tuple[str, object]]] = {}
    for t in np.arange(0, total - .001, 2):
        targets.setdefault(round(float(t) * FPS), []).append(("sheet", float(t)))
    for b in schedule:
        t = (b["start"] + b["end"]) / 2
        targets.setdefault(round(t * FPS), []).append(("beat", b["id"]))
    for row in narration:
        t = (row["start"] + row["end"]) / 2
        targets.setdefault(round(t * FPS), []).append(("caption", row))
    full_black, center_black, still_full, still_center, pts = [], [], [], [], []
    samples, beat_samples, previous = [], [], None
    caption_dir = out / "caption_frames"
    caption_dir.mkdir(parents=True, exist_ok=True)
    first_error, count = None, 0
    with av.open(str(video)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        metadata = dict(width=stream.width, height=stream.height,
                        nominal_fps=float(stream.average_rate or 0),
                        declared_frame_count=stream.frames, codec=stream.codec_context.name,
                        pixel_format=stream.codec_context.format.name)
        try:
            for i, fr in enumerate(c.decode(stream)):
                count = i + 1
                pts.append(float(fr.time) if fr.time is not None else None)
                gray = fr.reformat(width=320, height=180, format="gray").to_ndarray()
                center = gray[36:144, 80:240]
                full_black.append(bool(gray.mean() < 2.5 and np.percentile(gray, 99) < 7))
                center_black.append(bool(center.mean() < 2.5 and np.percentile(center, 99) < 7))
                if previous is None:
                    still_full.append(False)
                    still_center.append(False)
                else:
                    diff = np.abs(gray.astype(np.int16) - previous.astype(np.int16))
                    middle = diff[36:144, 80:240]
                    still_full.append(bool(diff.mean() < .10 and np.mean(diff <= 1) > .995))
                    still_center.append(bool(middle.mean() < .10 and np.mean(middle <= 1) > .995))
                previous = gray
                if i in targets:
                    rgb = Image.fromarray(fr.to_ndarray(format="rgb24"))
                    for purpose, key in targets[i]:
                        if purpose == "sheet":
                            samples.append((float(key), beat_at(schedule, float(key))["id"], rgb.copy()))
                        elif purpose == "beat":
                            beat_samples.append((i / FPS, str(key), rgb.copy()))
                        else:
                            try:
                                ref = caption_overlay(Image.new("RGB", (W, H), BG), .5, [(0, 1, key["caption"])])
                                ref_a, actual = np.asarray(ref), np.asarray(rgb)
                                glyph = ref_a.mean(axis=2) > 180
                                coverage = float(np.mean(actual.mean(axis=2)[glyph] > 160)) if np.any(glyph) else 0.0
                            except ValueError:
                                coverage = 0.0
                            key["decoded_caption_white_glyph_coverage"] = round(coverage, 5)
                            key["caption_checked_at"] = round(i / FPS, 6)
                            path = caption_dir / f"{key['id']}.jpg"
                            rgb.save(path, quality=90)
                            key["caption_frame"] = str(path)
                if count % 480 == 0:
                    print(f"Decoded {count} frames ({count/FPS:.1f}s)", flush=True)
        except Exception as exc:
            first_error = f"{type(exc).__name__}: {exc}"
    sheet(samples, out / "sheet.jpg", 6, 320)
    for i, start in enumerate(range(0, len(samples), 20), 1):
        sheet(samples[start:start + 20], out / f"sheet_{i}.jpg", 3, 640)
    sheet(beat_samples, out / "beat_sheet.jpg", 3, 640)
    black_events = [event(a, b, schedule, "full_black") for a, b in ranges(full_black)]
    black_events += [event(a, b, schedule, "center_black") for a, b in ranges(center_black)]
    # A freeze run consists of repeated transitions, so include its first frame.
    freeze_events = []
    for name, mask in (("whole_frame", still_full), ("center_panel", still_center)):
        for a, b in ranges(mask, math.floor(.7 * FPS)):
            row = event(max(0, a - 1), b, schedule, "freeze")
            if row["duration"] <= .7:
                continue
            row["region"] = name
            freeze_events.append(row)
    deltas = np.diff([p for p in pts if p is not None])
    metadata.update(decoded_frames=count, decoded_duration=round(count / FPS, 6),
                    first_pts=pts[0] if pts else None, last_pts=pts[-1] if pts else None,
                    pts_delta_min=float(deltas.min()) if deltas.size else None,
                    pts_delta_max=float(deltas.max()) if deltas.size else None,
                    decode_error=first_error, black_events=black_events, freezes=freeze_events,
                    sheet_sample_count=len(samples), beat_sample_count=len(beat_samples))
    return metadata


def voice_presence(audio: np.ndarray, voice: np.ndarray, at: float, name: str = "narration") -> dict:
    """Estimate an aligned known voice component in the final AAC, within 80 ms.

    Regression residual includes the original shot and codec error. It is an
    objective waveform-presence check, not an intelligibility score.
    """
    pad = round(.08 * SR)
    begin = round(at * SR)
    left, right = max(0, begin - pad), min(audio.size, begin + voice.size + pad)
    window = audio[left:right].astype(np.float64)
    v = voice.astype(np.float64)
    if window.size < v.size or not np.any(v):
        return dict(error="Source narration is silent or does not fit in output audio")
    n = 1 << (window.size + v.size - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(window, n) * np.fft.rfft(v[::-1], n), n)
    valid = corr[v.size - 1:window.size]
    offset = int(np.argmax(valid))
    aligned = window[offset:offset + v.size]
    scale = float(np.dot(aligned, v) / max(np.dot(v, v), 1e-12))
    residual = aligned - scale * v
    denom = math.sqrt(float(np.dot(aligned, aligned) * np.dot(v, v)))
    source_rms = math.sqrt(float(np.mean((scale * v) ** 2)))
    residual_rms = math.sqrt(float(np.mean(residual ** 2)))
    return dict(alignment_offset_seconds=round((left + offset - begin) / SR, 6),
                correlation=round(float(np.dot(aligned, v) / max(denom, 1e-12)), 6),
                **{f"estimated_{name}_scale": round(scale, 6),
                   f"{name}_to_residual_db": round(db(source_rms) - db(residual_rms), 3)},
                residual_rms_dbfs=db(residual_rms))


def author_voice_check(schedule: list[dict], narration: list[dict], audio: np.ndarray) -> list[dict]:
    """Find the author's recorded waveform in every proof, hook and Studio take.

    The reference includes the declared gain, loop and narration ducking. This
    lets the regression test the quiet final comparison fairly while a spoken
    explanation is mixed over it. Other sound, including narration, stays in
    the residual. It does not establish how understandable a voice sounds.
    """
    cache = {}

    def source(path):
        key = str(path)
        if key not in cache:
            cache[key] = read_audio(path, SR)
        return cache[key]

    total = round(schedule[-1]["end"] * SR)
    speech = np.zeros(total, np.float32)
    for row in narration:
        b = next(entry["beat"] for entry in schedule if entry["id"] == row["id"])
        v = source(row["source"]) * float(b.get("vo_gain", 1))
        at = round(row["start"] * SR)
        n = min(v.size, max(0, speech.size - at))
        speech[at:at + n] += v[:n]

    rows = []
    for entry in schedule:
        b, name = entry["beat"], entry["id"]
        if not (name in {"cold_open", "perform", "small_and_big"}
                or name.startswith("performance_") or name.startswith("comparison_")):
            continue
        n = round((entry["end"] - entry["start"]) * SR)
        if b["type"] == "compare":
            q = b["panels"][int(b.get("audio", 0))]
            path = q.get("audio_src", q["src"])
            begin = float(q.get("audio_from", q.get("start", 0)))
            gain = float(b.get("audio_gain", 1)) * float(q.get("audio_gain", 1))
            period = round(float(b.get("loop", b["dur"])) * SR)
        else:
            path = b.get("audio_src", b["src"])
            begin = float(b.get("audio_from", 0)) if b.get("audio_src") else float(b.get("start", 0))
            gain = float(b.get("audio_gain", 1)) if b.get("audio_src") else float(b.get("source_gain", 1))
            period = n
        raw = source(path)[round(begin * SR):round(begin * SR) + period]
        reference = np.zeros(period, np.float32)
        reference[:raw.size] = raw * gain
        if period < n:
            reference = np.tile(reference, math.ceil(n / period))[:n]
        offset = round(float(b.get("audio_at", 0)) * SR)
        if offset:
            reference = np.pad(reference, (offset, 0))[:n]
        at = round(entry["start"] * SR)
        voice = speech[at:at + n]
        active = np.flatnonzero(voice != 0)
        duck = np.ones(n, np.float32)
        if active.size:
            # The renderer smooths a rectangle from first to last speech sample
            # with a 150 ms box. Interval overlap equals that convolution.
            k = round(.15 * SR)
            indices = np.arange(n)
            left = np.maximum(active[0], indices - k // 2)
            right = np.minimum(active[-1] + 1, indices + (k - 1) // 2 + 1)
            speaking = np.maximum(0, right - left) / k
            duck = (1 - (1 - DUCK) * speaking).astype(np.float32)
        reference *= duck
        mix = audio[at:at + n]
        row = dict(id=name, source=str(path), source_from=begin, start=entry["start"], end=entry["end"],
                   declared_gain=gain, raw_voice_levels=levels(raw),
                   expected_voice_after_gain_and_duck_levels=levels(reference), mix_levels=levels(mix),
                   narration_ducks_voice=bool(active.size),
                   waveform_presence=voice_presence(audio, reference, entry["start"], "author_voice"))
        hop = round(.02 * SR)
        blocks = [float(np.sqrt(np.mean(reference[i:i + hop] ** 2))) for i in range(0, reference.size, hop)]
        threshold = max(blocks, default=0) * .08
        mask = np.repeat(np.asarray(blocks) > max(.001, threshold), hop)[:reference.size]
        row["active_voice_rms_dbfs"] = levels(reference[mask])["rms_dbfs"]
        if mix.size == reference.size:
            row["active_mix_rms_dbfs"] = levels(mix[mask])["rms_dbfs"]
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=RUN / "demo_v11.mp4")
    parser.add_argument("--storyboard", type=Path, default=Path(__file__).with_name("storyboard.json"))
    parser.add_argument("--output", type=Path, help="Defaults to the video's directory")
    parser.add_argument("--metadata-only", action="store_true", help="Check the storyboard and narration before rendering")
    args = parser.parse_args()
    out = args.output or args.video.parent
    out.mkdir(parents=True, exist_ok=True)
    board = json.loads(args.storyboard.read_text(encoding="utf-8"))
    beats = board["beats"] if isinstance(board, dict) else board
    schedule = timeline(beats)
    total = schedule[-1]["end"]
    narrations, errors = narration_check(schedule, total, out)
    report = dict(video=str(args.video), storyboard=str(args.storyboard), expected_duration=total,
                  expected_frames=sum(b["frames"] for b in schedule), narration=narrations,
                  errors=errors, review=[],
                  limitations="Waveform and pixel measurements do not replace listening or editorial review.")
    if total > 120:
        errors.append(f"Picture exceeds two minutes: {total:.6f}s")
    if not args.metadata_only:
        result = scan_video(args.video, schedule, narrations, out)
        report["picture"] = result
        if result["decode_error"]:
            errors.append(f"Video decode failed: {result['decode_error']}")
        if (result["width"], result["height"]) != (W, H):
            errors.append(f"Unexpected dimensions: {result['width']}x{result['height']}")
        if abs(result["nominal_fps"] - FPS) > 1e-6:
            errors.append(f"Unexpected frame rate: {result['nominal_fps']}")
        if result["decoded_frames"] != report["expected_frames"]:
            errors.append(f"Expected {report['expected_frames']} frames; decoded {result['decoded_frames']}")
        if result["pts_delta_min"] is not None and (abs(result["pts_delta_min"] - 1 / FPS) > 1e-5
                                                   or abs(result["pts_delta_max"] - 1 / FPS) > 1e-5):
            errors.append("Nonuniform frame presentation timestamps")
        for row in result["black_events"] + result["freezes"]:
            if row["classification"] == "review":
                report["review"].append(row)
        audio = read_audio(args.video, SR)
        report["audio"] = dict(duration=round(audio.size / SR, 6), levels=levels(audio), beats=[])
        if abs(audio.size / SR - total) > .08:
            errors.append(f"Audio/picture duration mismatch: {audio.size/SR:.6f}s vs {total:.6f}s")
        for entry in schedule:
            segment = audio[round(entry["start"] * SR):round(entry["end"] * SR)]
            report["audio"]["beats"].append(dict(id=entry["id"], start=entry["start"], end=entry["end"],
                                                 last_100ms_levels=levels(segment[-round(.1 * SR):]), **levels(segment)))
        for row in narrations:
            segment = audio[round(row["start"] * SR):round(row["end"] * SR)]
            row["mix_window_levels"] = levels(segment)
            b = next(b["beat"] for b in schedule if b["id"] == row["id"])
            voice = read_audio(row["source"], SR) * float(b.get("vo_gain", 1))
            row["waveform_presence"] = voice_presence(audio, voice, row["start"])
            if row.get("decoded_caption_white_glyph_coverage", 0) < .9:
                errors.append(f"Decoded caption coverage below 90%: {row['id']}")
            if row["waveform_presence"].get("estimated_narration_scale", 0) < .5:
                errors.append(f"Narration waveform weak or absent in mix: {row['id']}")
        report["author_voice"] = author_voice_check(schedule, narrations, audio)
        for row in report["author_voice"]:
            if row["waveform_presence"].get("estimated_author_voice_scale", 0) < .5:
                errors.append(f"Author voice waveform weak or absent in mix: {row['id']}")
        if report["audio"]["levels"]["near_full_scale_samples"]:
            report["review"].append(dict(kind="audio_near_full_scale", **report["audio"]["levels"]))
    report["status"] = "errors" if errors else "review" if report["review"] else "measured_checks_passed"
    destination = out / ("qa_metadata.json" if args.metadata_only else "qa.json")
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {total:.3f}s, {report['expected_frames']} expected frames")
    print(f"{len(errors)} errors; {len(report['review'])} review items -> {destination}")
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
