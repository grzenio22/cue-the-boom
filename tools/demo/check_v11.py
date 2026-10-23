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


RUN = Path("D:/openHiggsfield/runs/demo_v13")


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
    manifests = {}

    def manifest(path: Path) -> dict:
        if path not in manifests:
            manifests[path] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return manifests[path]

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
        written = manifest(path.parent.parent / "vo_lines.json")
        declared = manifest(path.parent / "durations.json")
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
        if active.size and row["last_active_sample_before_file_end_seconds"] < .02:
            issues.append(f"Narration source ends without 20 ms of quiet padding: {entry['id']}")
        if not text:
            issues.append(f"Narration lacks a caption: {entry['id']}")
        else:
            if path.stem in written:
                row["caption_matches_spoken_text"] = pronunciation_key(text) == pronunciation_key(written[path.stem])
                if not row["caption_matches_spoken_text"]:
                    issues.append(f"Caption differs from narration input: {entry['id']}")
            else:
                issues.append(f"Narration text missing from input manifest: {path.stem}")
        segments = b.get("vo_captions", [[0, audio.size / SR, text]])
        row["captions"] = []
        previous_caption_end = 0.0
        captioned = np.zeros(audio.size, dtype=bool)
        for i, segment in enumerate(segments):
            if not isinstance(segment, (list, tuple)) or len(segment) != 3:
                issues.append(f"Invalid caption segment {i + 1}: {entry['id']}")
                continue
            relative_begin, relative_end, caption = float(segment[0]), float(segment[1]), segment[2]
            c = dict(id=f"{entry['id']}_{i+1}", caption=caption,
                     start=round(begin + relative_begin, 6), end=round(begin + relative_end, 6),
                     source_begin=relative_begin, source_end=relative_end)
            if (not isinstance(caption, str) or not caption.strip() or relative_begin < -.002
                    or relative_end <= relative_begin or relative_end > audio.size / SR + .002):
                issues.append(f"Caption segment outside narration or empty: {c['id']}")
            if relative_begin < previous_caption_end - .002:
                issues.append(f"Overlapping caption segments: {c['id']}")
            previous_caption_end = relative_end
            captioned[max(0, round(relative_begin * SR)):max(0, round(relative_end * SR))] = True
            try:
                caption_overlay(blank.copy(), .5, [(0, 1, caption)])
                c["caption_fits_two_line_safe_area"] = True
            except (ValueError, AttributeError) as exc:
                c["caption_fits_two_line_safe_area"] = False
                issues.append(f"{c['id']}: {exc}")
            row["captions"].append(c)
        caption_text = " ".join(str(c["caption"]) for c in row["captions"])
        row["segmented_captions_match_full_transcript"] = pronunciation_key(caption_text) == pronunciation_key(text)
        if not row["segmented_captions_match_full_transcript"]:
            issues.append(f"Caption segments omit or change narration words: {entry['id']}")
        row["uncaptioned_active_audio_seconds"] = round(float(np.count_nonzero(~captioned[active])) / SR, 6)
        if row["uncaptioned_active_audio_seconds"] > .04:
            issues.append(f"Narration audio plays outside caption windows: {entry['id']}")
        row["caption_fits_two_line_safe_area"] = all(c["caption_fits_two_line_safe_area"] for c in row["captions"])
        if path.stem in declared:
            row["duration_manifest_error_seconds"] = round(audio.size / SR - declared[path.stem], 6)
            if abs(row["duration_manifest_error_seconds"]) > .002:
                issues.append(f"Narration duration manifest differs: {path.stem}")
        else:
            issues.append(f"Narration duration missing from manifest: {path.stem}")
        if begin < previous_end - .002:
            issues.append(f"Overlapping narration at {begin:.3f}s: {entry['id']}")
        if end > total + .002:
            issues.append(f"Narration cut at picture end: {entry['id']} ({end-total:.3f}s)")
        previous_end = max(previous_end, end)
        result.append(row)
    return result, issues


def source_bounds_check(schedule: list[dict]) -> tuple[list[dict], list[str]]:
    """Check selected source windows before the renderer can silently hold or pad them."""
    result, issues, durations = [], [], {}

    def check(path, start, duration, entry, media, hold=False):
        p = Path(path)
        row = dict(id=entry["id"], source=str(p), media=media,
                   source_from=round(start, 6), requested_end=round(start + duration, 6))
        key = (str(p), media)
        if not p.is_file():
            issues.append(f"Missing {media} source: {p}")
            result.append(row)
            return
        if key not in durations:
            if media == "audio":
                durations[key] = read_audio(p, SR).size / SR
            else:
                with av.open(str(p)) as c:
                    stream = c.streams.video[0]
                    durations[key] = (float(stream.duration * stream.time_base) if stream.duration is not None
                                      else float(c.duration / av.time_base) if c.duration is not None else None)
        available = durations[key]
        row["available_duration"] = available
        if start < 0:
            issues.append(f"Negative {media} source start: {entry['id']}")
        if available is not None and start + duration > available + 1 / FPS:
            if hold:
                row["classification"] = "designed_tail_hold" if media == "video" else "designed_silent_tail"
                row["tail_duration"] = round(start + duration - available, 6)
                row["reason"] = entry["beat"].get("audio_tail_reason" if media == "audio" else "hold_reason",
                                                "Designed source tail")
            else:
                issues.append(f"{media.title()} source ends before selected window: {entry['id']} "
                              f"({start+duration:.3f}s > {available:.3f}s)")
        result.append(row)

    for entry in schedule:
        b, duration = entry["beat"], entry["end"] - entry["start"]
        kind = b["type"]
        if kind == "clip":
            speed = float(b.get("speed", 1))
            if b.get("hold_tail") and not b.get("hold_reason"):
                issues.append(f"Source hold lacks an editorial reason: {entry['id']}")
            check(b["src"], float(b.get("start", 0)), duration * speed, entry, "video",
                  entry["id"] == "cold_open" or bool(b.get("hold_tail") and b.get("hold_reason")))
            if b.get("audio_src"):
                if b.get("audio_tail_silence") and not b.get("audio_tail_reason"):
                    issues.append(f"Audio silent tail lacks an editorial reason: {entry['id']}")
                check(b["audio_src"], float(b.get("audio_from", 0)),
                      max(0, duration - float(b.get("audio_at", 0))), entry, "audio",
                      entry["id"] == "cold_open" or bool(b.get("audio_tail_silence") and b.get("audio_tail_reason")))
        elif kind == "triptych":
            for src in b["srcs"]:
                check(src, float(b.get("start", 0)), duration * float(b.get("speed", 1)), entry, "video")
        elif kind in {"turns", "compare"}:
            summed_duration = 0
            for q in b["panels"]:
                period = float(q["dur"]) if kind == "turns" else float(b.get("loop", b["dur"]))
                check(q["src"], float(q.get("start", 0)), period, entry, "video")
                if q.get("audio_src"):
                    check(q["audio_src"], float(q.get("audio_from", q.get("start", 0))), period, entry, "audio")
                summed_duration += round(period * FPS) / FPS
            if kind == "turns" and abs(summed_duration - duration) > .002:
                issues.append(f"Turn durations differ from beat duration: {entry['id']}")
        elif kind == "image":
            if not Path(b["src"]).is_file():
                issues.append(f"Missing card: {b['src']}")
        if kind != "clip" and b.get("audio_src"):
            check(b["audio_src"], float(b.get("audio_from", 0)), duration, entry, "audio", hold=True)
        if b.get("element_src"):
            if not Path(b["element_src"]).is_file():
                issues.append(f"Missing proof element: {b['element_src']}")
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
    if b.get("hold_tail") and b.get("hold_reason"):
        with av.open(str(b["src"])) as container:
            stream = container.streams.video[0]
            duration = float(stream.duration * stream.time_base)
        hold_at = entry["start"] + (duration - float(b.get("start", 0))) / float(b.get("speed", 1))
        if start >= hold_at - 1 / FPS - .001:
            return b["hold_reason"]
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
    elif kind == "center_black" and (entry["id"] in {"one_generation", "how", "raw_effect", "title_raw"}
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
        for caption in row["captions"]:
            t = (caption["start"] + caption["end"]) / 2
            targets.setdefault(round(t * FPS), []).append(("caption", caption))
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

    def reference(path, begin, period, n, gain):
        raw = source(path)[round(begin * SR):round(begin * SR) + period]
        ref = np.zeros(period, np.float32)
        ref[:raw.size] = raw * gain
        return raw, np.tile(ref, math.ceil(n / max(1, period)))[:n]

    def measure(name, sources, ref, raw, start, end):
        n, at = ref.size, round(start * SR)
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
        ref *= duck
        mix = audio[at:at + n]
        row = dict(id=name, sources=sources, start=start, end=end, raw_voice_levels=levels(raw),
                   expected_voice_after_gain_and_duck_levels=levels(ref), mix_levels=levels(mix),
                   narration_ducks_voice=bool(active.size),
                   waveform_presence=voice_presence(audio, ref, start, "author_voice"))
        hop = round(.02 * SR)
        blocks = [float(np.sqrt(np.mean(ref[i:i + hop] ** 2))) for i in range(0, ref.size, hop)]
        threshold = max(blocks, default=0) * .08
        mask = np.repeat(np.asarray(blocks) > max(.001, threshold), hop)[:ref.size]
        row["active_voice_rms_dbfs"] = levels(ref[mask])["rms_dbfs"]
        if mix.size == ref.size:
            row["active_mix_rms_dbfs"] = levels(mix[mask])["rms_dbfs"]
        return row

    rows = []
    for entry in schedule:
        b, name = entry["beat"], entry["id"]
        if not (b.get("author_voice") or name in {"cold_open", "perform", "small_and_big"} or b["type"] in {"turns", "compare"}
                or name.startswith("performance_") or name.startswith("comparison_")):
            continue
        n = round((entry["end"] - entry["start"]) * SR)
        if b["type"] == "turns":
            offset = 0
            for k, q in enumerate(b["panels"]):
                period = round(float(q["dur"]) * FPS) * (SR // FPS)
                path = q.get("audio_src", q["src"])
                begin = float(q.get("audio_from", q.get("start", 0)))
                gain = float(b.get("audio_gain", 1)) * float(q.get("audio_gain", 1))
                raw, ref = reference(path, begin, period, period, gain)
                start = entry["start"] + offset / SR
                rows.append(measure(f"{name}_panel_{k+1}", [dict(source=str(path), source_from=begin,
                                    declared_gain=gain)], ref, raw, start, start + period / SR))
                offset += period
        elif b["type"] == "compare":
            selected = b.get("audio", 0)
            panels = b["panels"] if selected == "mix" else [b["panels"][int(selected)]]
            refs, raw_parts, sources = [], [], []
            for q in panels:
                path = q.get("audio_src", q["src"])
                begin = float(q.get("audio_from", q.get("start", 0)))
                gain = float(b.get("audio_gain", 1)) * float(q.get("audio_gain", 1)) / len(panels)
                period = round(float(b.get("loop", b["dur"])) * FPS) * (SR // FPS)
                raw, ref = reference(path, begin, period, n, gain)
                refs.append(ref)
                raw_parts.append(raw)
                sources.append(dict(source=str(path), source_from=begin, declared_gain=gain))
            rows.append(measure(name, sources, np.sum(refs, axis=0), np.concatenate(raw_parts),
                                entry["start"], entry["end"]))
        else:
            path = b.get("audio_src", b["src"])
            begin = float(b.get("audio_from", 0)) if b.get("audio_src") else float(b.get("start", 0))
            gain = float(b.get("audio_gain", 1)) if b.get("audio_src") else float(b.get("source_gain", 1))
            raw, ref = reference(path, begin, n, n, gain)
            offset = round(float(b.get("audio_at", 0)) * SR)
            if offset:
                ref = np.pad(ref, (offset, 0))[:n]
            rows.append(measure(name, [dict(source=str(path), source_from=begin, declared_gain=gain)],
                                ref, raw, entry["start"], entry["end"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=RUN / "demo_v13.mp4")
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
    source_windows, source_errors = source_bounds_check(schedule)
    errors.extend(source_errors)
    report = dict(video=str(args.video), storyboard=str(args.storyboard), expected_duration=total,
                  expected_frames=sum(b["frames"] for b in schedule), narration=narrations,
                  source_windows=source_windows,
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
            if row["kind"] == "full_black":
                errors.append(f"Whole black frames at {row['start']:.3f}s–{row['end']:.3f}s ({row['beat']})")
            elif row["classification"] == "review":
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
            for caption in row["captions"]:
                if caption.get("decoded_caption_white_glyph_coverage", 0) < .9:
                    errors.append(f"Decoded caption coverage below 90%: {caption['id']}")
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
