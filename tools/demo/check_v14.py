"""Verify the eight-second recap and preservation of the approved previous cut.

The storyboard comparison is exact outside the recap. Every decoded picture
outside it is compared at the corresponding frame in the previous cut. Only
the retimed narration's subtitle band is excluded during the unchanged reprise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cuesync" / "src"))
from cuesync.media import read_audio  # noqa: E402


ROOT = Path("D:/openHiggsfield")
RUN = ROOT / "runs/demo_v14"
FPS, SR = 24, 48000
REPRISE = "innovation_hits"
COMPARISON_SIZE = (960, 540)
CAPTION_TOP = 832


def beats_of(board: dict | list) -> list[dict]:
    return board["beats"] if isinstance(board, dict) else board


def schedule(beats: list[dict]) -> list[dict]:
    elapsed, result = 0, []
    for beat in beats:
        count = round(float(beat["dur"]) * FPS)
        result.append(dict(id=beat["id"], start_frame=elapsed, end_frame=elapsed + count,
                           start=elapsed / FPS, end=(elapsed + count) / FPS, beat=beat))
        elapsed += count
    return result


def recap_bounds(beats: list[dict]) -> tuple[int, int]:
    ids = [b["id"] for b in beats]
    return ids.index("innovation"), ids.index(REPRISE)


def board_structure(board: dict | list, base: dict | list) -> tuple[dict, list[str]]:
    beats, prior = beats_of(board), beats_of(base)
    errors = []
    try:
        begin, end = recap_bounds(beats)
        old_begin, old_end = recap_bounds(prior)
    except ValueError as exc:
        return {}, [f"Missing recap or reprise boundary: {exc}"]
    if end <= begin or old_end <= old_begin:
        return {}, ["Recap and reprise boundaries are out of order"]
    if len({b["id"] for b in beats}) != len(beats):
        errors.append("Duplicate beat identifiers in current storyboard")
    before, old_before = beats[:begin], prior[:old_begin]
    after, old_after = beats[end:], prior[old_end:]
    if [b["id"] for b in before] != [b["id"] for b in old_before]:
        errors.append("Beat order before recap changed")
    if [b["id"] for b in after] != [b["id"] for b in old_after]:
        errors.append("Beat order after recap changed")
    old_by_id = {b["id"]: b for b in old_before + old_after}
    differences, unchanged = [], []
    for beat in before + after:
        old = old_by_id.get(beat["id"])
        if beat != old:
            keys = sorted(k for k in set(beat) | set(old or {}) if beat.get(k) != (old or {}).get(k))
            differences.append(dict(id=beat["id"], fields=keys))
            errors.append(f"Untouched beat differs: {beat['id']} ({', '.join(keys)})")
        else:
            unchanged.append(beat["id"])
    current, original = schedule(beats), schedule(prior)
    recap_start, recap_end = current[begin]["start_frame"], current[end]["start_frame"]
    old_start, old_finish = original[old_begin]["start_frame"], original[old_end]["start_frame"]
    if recap_start != old_start:
        errors.append("Recap start moved")
    if recap_end - recap_start != 8 * FPS:
        errors.append(f"Recap must be 8.000 seconds; found {(recap_end-recap_start)/FPS:.6f}")
    removed = (old_finish - old_start) - (recap_end - recap_start)
    if original[-1]["end_frame"] - current[-1]["end_frame"] != removed:
        errors.append("Total shortening differs from recap shortening")
    result = dict(unchanged_beat_count=len(unchanged), unchanged_beats=unchanged,
                  changed_outside_recap=differences, recap_beats=[b["id"] for b in beats[begin:end]],
                  recap_start_frame=recap_start, recap_end_frame=recap_end,
                  recap_start=recap_start/FPS, recap_end=recap_end/FPS,
                  recap_duration=(recap_end-recap_start)/FPS,
                  base_recap_start_frame=old_start, base_recap_end_frame=old_finish,
                  base_recap_start=old_start/FPS, base_recap_end=old_finish/FPS,
                  base_recap_duration=(old_finish-old_start)/FPS,
                  removed_frames=removed, removed_seconds=removed/FPS,
                  expected_frames=current[-1]["end_frame"], duration=current[-1]["end"],
                  base_expected_frames=original[-1]["end_frame"], base_duration=original[-1]["end"],
                  reprise_start=current[end]["start"], reprise_end=current[end]["end"],
                  reprise_start_frame=current[end]["start_frame"], reprise_end_frame=current[end]["end_frame"],
                  reprise_dictionary_unchanged=beats[end] == prior[old_end])
    return result, errors


def text_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def voice_metadata(board: dict | list, base: dict | list) -> tuple[dict, list[str]]:
    errors, rows, previous_end, contribution = [], [], 0.0, None
    prior = next(b for b in beats_of(base) if b["id"] == "innovation")
    for entry in schedule(beats_of(board)):
        beat = entry["beat"]
        if not beat.get("vo"):
            continue
        path = Path(beat["vo"])
        if not path.is_file():
            errors.append(f"Missing narration source: {path}")
            continue
        voice = read_audio(path, SR)
        begin = entry["start"] + float(beat.get("vo_at", .15))
        end = begin + voice.size / SR
        row = dict(id=beat["id"], source=str(path), start=begin, end=end, duration=voice.size/SR)
        rows.append(row)
        if begin < previous_end - .002:
            errors.append(f"Narration overlaps at {beat['id']}")
        previous_end = max(end, previous_end)
        if beat["id"] == "innovation":
            contribution = row
            if beat.get("vo_text") != prior.get("vo_text"):
                errors.append("Contribution transcript differs from the complete original line")
            text = beat.get("vo_text", "")
            segments = beat.get("vo_captions", [])
            if text_key(" ".join(str(c[2]) for c in segments)) != text_key(text):
                errors.append("Contribution caption phrases omit or change words")
            captioned = np.zeros(voice.size, bool)
            previous_caption_end = 0.0
            for left, right, phrase in segments:
                if left < previous_caption_end - .002 or left < 0 or right <= left or right > voice.size/SR + .002:
                    errors.append("Contribution caption timing is invalid or overlapping")
                previous_caption_end = right
                captioned[max(0, round(left*SR)):min(voice.size, round(right*SR))] = True
            threshold = max(.001, float(np.max(np.abs(voice))) * .01) if voice.size else .001
            active = np.flatnonzero(np.abs(voice) > threshold)
            uncovered = np.count_nonzero(~captioned[active]) / SR
            padding = (voice.size - active[-1] - 1)/SR if active.size else 0
            row.update(transcript=text, caption_phrases=segments, uncaptioned_active_seconds=uncovered,
                       trailing_quiet_seconds=padding, full_transcript_retained=text == prior.get("vo_text"))
            if uncovered > .04 or padding < .02 or not active.size:
                errors.append("Contribution has uncovered speech, no speech, or insufficient quiet ending")
            original_voice = read_audio(prior["vo"], SR)
            row["original_duration"] = original_voice.size / SR
            row["duration_ratio"] = original_voice.size / max(1, voice.size)
            try:
                lines_path = path.parent.parent / "vo_lines.json"
                durations_path = path.parent / "durations.json"
                lines = json.loads(lines_path.read_text(encoding="utf-8"))
                durations = json.loads(durations_path.read_text(encoding="utf-8"))
                row["transcript_manifest"] = str(lines_path)
                row["duration_manifest"] = str(durations_path)
                if text_key(lines.get(path.stem, "")) != text_key(prior.get("vo_text", "")):
                    errors.append("Retimed source transcript manifest differs from original complete line")
                if abs(float(durations[path.stem]) - voice.size/SR) > .002:
                    errors.append("Retimed source duration manifest differs from decoded WAV")
                provenance_path = path.parent / "provenance.json"
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                row["retiming_provenance"] = provenance
                original_path = Path(prior["vo"])
                expected_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()
                if (Path(provenance.get("source", "")) != original_path
                        or provenance.get("source_sha256") != expected_hash):
                    errors.append("Retiming provenance does not match the complete original source file")
                if (provenance.get("words_removed") != 0 or not provenance.get("pitch_preserved")
                        or text_key(provenance.get("complete_transcript", "")) != text_key(prior.get("vo_text", ""))):
                    errors.append("Retiming provenance does not retain the complete line with pitch preserved")
                tempo = float(provenance.get("tempo", 0))
                if abs(tempo - row["duration_ratio"]) > .005 or abs(tempo - 1.28) > .005:
                    errors.append("Retimed duration does not match the declared 1.28 tempo")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"Retimed source manifests are missing or invalid: {exc}")
            if path == Path(prior["vo"]):
                errors.append("Contribution still refers to the original unretimed source")
    total = schedule(beats_of(board))[-1]["end"]
    if previous_end > total + .002:
        errors.append("Narration ends after picture")
    if contribution is None:
        errors.append("Contribution narration is missing")
    return dict(lines=rows, contribution=contribution), errors


def waveform_presence(audio: np.ndarray, voice: np.ndarray, at: float) -> dict:
    pad, begin = round(.08*SR), round(at*SR)
    left, right = max(0, begin-pad), min(audio.size, begin+voice.size+pad)
    window, target = audio[left:right].astype(np.float64), voice.astype(np.float64)
    if window.size < target.size or not np.any(target):
        return dict(error="Complete narration does not fit in final audio")
    n = 1 << (window.size+target.size-1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(window, n)*np.fft.rfft(target[::-1], n), n)
    offset = int(np.argmax(corr[target.size-1:window.size]))
    aligned = window[offset:offset+target.size]
    dot = float(np.dot(aligned, target))
    scale = dot/max(float(np.dot(target, target)), 1e-12)
    correlation = dot/max(math.sqrt(float(np.dot(aligned, aligned)*np.dot(target, target))), 1e-12)
    return dict(offset_seconds=(left+offset-begin)/SR, scale=scale, correlation=correlation,
                samples_checked=target.size, duration_checked=target.size/SR)


def decoded_frames(path: Path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            yield index, frame


def new_metrics() -> dict:
    return dict(frames=0, sample_count=0, absolute_sum=0.0, square_sum=0.0,
                maximum_pixel_error=0, worst_frame_mean_error=0.0, worst_frames=[])


def record(metrics: dict, difference: np.ndarray, current_index: int, base_index: int) -> None:
    absolute = np.abs(difference)
    mean = float(np.mean(absolute))
    metrics["frames"] += 1
    metrics["sample_count"] += difference.size
    metrics["absolute_sum"] += float(np.sum(absolute, dtype=np.float64))
    metrics["square_sum"] += float(np.sum(difference.astype(np.float32)**2, dtype=np.float64))
    metrics["maximum_pixel_error"] = max(metrics["maximum_pixel_error"], int(np.max(absolute)))
    metrics["worst_frame_mean_error"] = max(metrics["worst_frame_mean_error"], mean)
    metrics["worst_frames"].append(dict(frame=current_index, time=current_index/FPS,
                                         base_frame=base_index, mean_absolute_error=round(mean, 6)))
    metrics["worst_frames"] = sorted(metrics["worst_frames"], key=lambda row: row["mean_absolute_error"], reverse=True)[:5]


def finish_metrics(metrics: dict) -> dict:
    count = max(1, metrics.pop("sample_count"))
    mean = metrics.pop("absolute_sum")/count
    mse = metrics.pop("square_sum")/count
    metrics.update(mean_absolute_error=round(mean, 6), mean_square_error=round(mse, 6),
                   psnr_db=round(10*math.log10(255**2/mse), 5) if mse else None,
                   identical=not bool(mse))
    return metrics


def picture_preservation(video: Path, base: Path, structure: dict) -> tuple[dict, list[str]]:
    errors, metrics = [], {name: new_metrics() for name in ("before_recap", "reprise_above_captions", "after_reprise")}
    baseline = iter(decoded_frames(base))
    old_index, old_frame = -1, None
    count = 0
    dimensions = set()
    for index, frame in decoded_frames(video):
        count = index+1
        dimensions.add((frame.width, frame.height))
        if structure["recap_start_frame"] <= index < structure["recap_end_frame"]:
            continue
        wanted = index if index < structure["recap_start_frame"] else index+structure["removed_frames"]
        try:
            while old_index < wanted:
                old_index, old_frame = next(baseline)
        except StopIteration:
            errors.append(f"Previous video ended before corresponding frame {wanted}")
            break
        current = frame.reformat(width=COMPARISON_SIZE[0], height=COMPARISON_SIZE[1], format="rgb24").to_ndarray()
        reference = old_frame.reformat(width=COMPARISON_SIZE[0], height=COMPARISON_SIZE[1], format="rgb24").to_ndarray()
        if index < structure["recap_start_frame"]:
            group = "before_recap"
        elif index < structure["reprise_end_frame"]:
            group = "reprise_above_captions"
            top = CAPTION_TOP*COMPARISON_SIZE[1]//1080
            current, reference = current[:top], reference[:top]
        else:
            group = "after_reprise"
        record(metrics[group], current.astype(np.int16)-reference.astype(np.int16), index, wanted)
        if count % 480 == 0:
            print(f"Compared preserved picture through {count/FPS:.1f}s", flush=True)
    for old_index, old_frame in baseline:
        pass
    if count != structure["expected_frames"]:
        errors.append(f"Current decoded frame count differs: {count} != {structure['expected_frames']}")
    if old_index+1 != structure["base_expected_frames"]:
        errors.append(f"Previous decoded frame count differs: {old_index+1} != {structure['base_expected_frames']}")
    if dimensions != {(1920, 1080)}:
        errors.append(f"Unexpected current decoded dimensions: {sorted(dimensions)}")
    finished = {key: finish_metrics(value) for key, value in metrics.items()}
    for key, item in finished.items():
        if not item["frames"]:
            errors.append(f"No preserved frames compared in {key}")
        if item["mean_absolute_error"] > 2 or item["worst_frame_mean_error"] > 4 or (item["psnr_db"] is not None and item["psnr_db"] < 32):
            errors.append(f"Preserved picture differs beyond encoding tolerance in {key}")
    return dict(decoded_frames=count, base_decoded_frames=old_index+1, dimensions=sorted(dimensions),
                comparison_dimensions=COMPARISON_SIZE, compared_frames=sum(v["frames"] for v in finished.values()),
                comparison="Every non-recap decoded frame, matched by frame index with the recap reduction offset",
                reprise_caption_exception=dict(top=CAPTION_TOP, bottom=1080,
                    reason="Complete narration and its captions were retimed; unchanged reprise picture is checked above this band"),
                thresholds=dict(maximum_mean_absolute_error=2, maximum_worst_frame_mean_error=4, minimum_psnr_db=32),
                groups=finished), errors


def audio_preservation(current: np.ndarray, base: np.ndarray, structure: dict) -> tuple[dict, list[str]]:
    """Compare the unchanged soundtrack at exact PCM offsets after AAC decoding."""
    rows, errors = [], []
    intervals = (("before_recap", 0.0, structure["recap_start"], 0.0),
                 ("after_reprise", structure["reprise_end"], structure["duration"], structure["removed_seconds"]))
    for name, begin, end, shift in intervals:
        left, right = round(begin*SR), round(end*SR)
        old_left, old_right = round((begin+shift)*SR), round((end+shift)*SR)
        actual = current[left:right].astype(np.float64)
        expected = base[old_left:old_right].astype(np.float64)
        if actual.size != right-left or expected.size != actual.size:
            errors.append(f"Complete preserved audio window does not fit: {name}")
            continue
        error = actual-expected
        source_rms = math.sqrt(float(np.mean(expected**2)))
        error_rms = math.sqrt(float(np.mean(error**2)))
        dot = float(np.dot(actual, expected))
        expected_energy = float(np.dot(expected, expected))
        correlation = dot/max(math.sqrt(float(np.dot(actual, actual))*expected_energy), 1e-12)
        scale = dot/max(expected_energy, 1e-12)
        row = dict(id=name, start=begin, end=end, base_start=begin+shift, base_end=end+shift,
                   sample_offset=round(shift*SR), compared_samples=actual.size,
                   expected_rms=source_rms, rms_error=error_rms,
                   rms_error_dbfs=20*math.log10(max(error_rms, 1e-12)),
                   residual_to_source_db=20*math.log10(max(error_rms, 1e-12)/max(source_rms, 1e-12)),
                   mean_absolute_error=float(np.mean(np.abs(error))), peak_error=float(np.max(np.abs(error))),
                   correlation=correlation, fitted_scale=scale)
        rows.append(row)
        if error_rms > .01 or (source_rms > .001 and (correlation < .98 or abs(scale-1) > .02)):
            errors.append(f"Preserved audio differs beyond AAC tolerance: {name}")
    return dict(sample_rate=SR, comparison="Decoded PCM at exact sample offsets; recap and its narrated reprise are excluded",
                thresholds=dict(maximum_rms_error=.01, minimum_correlation=.98, maximum_scale_error=.02),
                groups=rows), errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=RUN / "demo_v14.mp4")
    parser.add_argument("--base", type=Path, default=ROOT / "runs/demo_v13/demo_v13.mp4")
    parser.add_argument("--storyboard", type=Path, default=Path(__file__).with_name("storyboard.json"))
    parser.add_argument("--base-storyboard", type=Path, default=ROOT / "runs/demo_v13/storyboard.json")
    parser.add_argument("--output", type=Path, default=RUN / "preservation_check.json")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    board = json.loads(args.storyboard.read_text(encoding="utf-8"))
    base_board = json.loads(args.base_storyboard.read_text(encoding="utf-8"))
    structure, errors = board_structure(board, base_board)
    voice, voice_errors = voice_metadata(board, base_board)
    errors.extend(voice_errors)
    report = dict(video=str(args.video), base=str(args.base), storyboard=str(args.storyboard),
                  base_storyboard=str(args.base_storyboard), metadata_only=args.metadata_only,
                  structure=structure, narration=voice, errors=errors)
    if not args.metadata_only and not errors:
        picture, picture_errors = picture_preservation(args.video, args.base, structure)
        report["picture"] = picture
        errors.extend(picture_errors)
        contribution = voice["contribution"]
        beat = next(b for b in beats_of(board) if b["id"] == "innovation")
        waveform = read_audio(beat["vo"], SR)*float(beat.get("vo_gain", 1))
        final_audio = read_audio(args.video, SR)
        presence = waveform_presence(final_audio, waveform, contribution["start"])
        report["complete_contribution_waveform_presence"] = presence
        if presence.get("scale", 0) < .8 or presence.get("correlation", 0) < .8 or abs(presence.get("offset_seconds", 1)) > .04:
            errors.append("Complete retimed contribution waveform is weak, absent, or misaligned")
        sound, sound_errors = audio_preservation(final_audio, read_audio(args.base, SR), structure)
        report["audio_preservation"] = sound
        errors.extend(sound_errors)
    report["status"] = "errors" if errors else "metadata_passed" if args.metadata_only else "preservation_checks_passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"{report['status']}: {len(errors)} errors -> {args.output}")
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
