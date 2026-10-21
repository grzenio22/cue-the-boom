"""Make v12's synthetic narration with the established Kokoro-82M voice.

    python tools/demo/make_v11_voice.py

The normal project Python launches the existing narration environment. Input is
runs/demo_v12/vo_lines.json ({"name": "spoken text", ...}); outputs are 24 kHz
mono WAVs and durations.json in runs/demo_v12/vo12. The default voice and speed
match runs/round10/tts.py, which made the earlier narration. Cached weights are
used offline, and temporary files stay on D:.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import inspect
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys


RUN = Path("D:/openHiggsfield/runs/demo_v12")
TTS_PYTHON = Path("D:/openHiggsfield/runs/tts-venv/Scripts/python.exe")
TEMP = Path("D:/openHiggsfield_tmp/sol12/demo/tts")
CAPTION_PHRASES = {
    "contribution15": [
        "My contribution: separating generation from performance.",
        "Your voice plays one LTX instrument,",
        "without regenerating.",
    ],
    "new12": [
        "My contribution is separating generation from performance:",
        "one LTX effect from ComfyUI becomes a reusable instrument.",
        "Your voice controls when it hits and how big it plays,",
        "without another AI generation.",
    ],
}


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def caption_phrases(name: str, spoken: str, duration: float, tokens: list[dict]) -> list[list]:
    """Cover the WAV with complete phrases, changing captions between words."""
    phrases = CAPTION_PHRASES.get(name, [spoken])

    def letters(text):
        return re.sub(r"[^\w]", "", text, flags=re.UNICODE).casefold()

    if letters(" ".join(phrases)) != letters(spoken):
        raise ValueError(f"Caption phrases do not cover the transcript for {name}.")
    if len(phrases) == 1:
        return [[0.0, duration, phrases[0]]]
    words = [t for t in tokens if letters(t["text"])]
    if (letters(" ".join(t["text"] for t in words)) != letters(spoken)
            or any(t["start"] is None or t["end"] is None for t in words)):
        raise ValueError(f"Incomplete word timestamps for {name}.")
    boundaries = [0.0]
    character_end = 0
    word_end = 0
    for phrase in phrases[:-1]:
        character_end += len(letters(phrase))
        consumed = 0
        for index, word in enumerate(words):
            consumed += len(letters(word["text"]))
            if consumed >= character_end:
                if consumed != character_end or index + 1 == len(words):
                    raise ValueError(f"Caption change falls inside a token for {name}.")
                word_end = index
                break
        left, right = words[word_end], words[word_end + 1]
        if left["end"] > right["start"]:
            raise ValueError(f"Overlapping word timestamps at caption change for {name}.")
        boundary = round((left["end"] + right["start"]) / 2, 5)
        if not boundaries[-1] < boundary < duration:
            raise ValueError(f"Invalid caption timing for {name}.")
        boundaries.append(boundary)
    boundaries.append(duration)
    return [[boundaries[i], boundaries[i + 1], phrase] for i, phrase in enumerate(phrases)]


def package_names() -> dict[str, list[str]]:
    """Read installed wheel metadata without statting every file in large wheels.

    The narration environment's dependency scanner otherwise spends minutes
    checking all of torch's files before importing the speech model. This uses
    the same declared package names, or infers them from the wheel's RECORD.
    """
    found: dict[str, list[str]] = {}
    for dist in importlib.metadata.distributions():
        names = set((dist.read_text("top_level.txt") or "").split())
        if not names:
            for row in csv.reader(io.StringIO(dist.read_text("RECORD") or "")):
                filename = row[0].replace("\\", "/")
                name = filename.split("/")[0] if "/" in filename else inspect.getmodulename(filename)
                if name and "." not in name:
                    names.add(name)
        distribution_name = dist.metadata["Name"]
        for name in names:
            found.setdefault(name, []).append(distribution_name)
    return found


class ModelEntries:
    """Keep ALBERT and spaCy's installed tokenizer dependencies on import."""

    def __init__(self, entries):
        self.entries = entries

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            entry = next(self.entries)
            if entry.name in {"albert", "auto", "bert", "roberta", "xlm_roberta", "camembert", "bert_japanese", "__init__.py"}:
                return entry

    def close(self):
        self.entries.close()


def generate(lines_path: Path, output: Path, voice: str, speed: float, only: list[str] | None = None) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    original = importlib.metadata.packages_distributions
    original_scan = os.scandir

    def model_scan(path):
        entries = original_scan(path)
        p = Path(path)
        return ModelEntries(entries) if p.name == "models" and p.parent.name == "transformers" else entries

    importlib.metadata.packages_distributions = package_names
    os.scandir = model_scan
    try:
        from kokoro import KPipeline
    finally:
        importlib.metadata.packages_distributions = original
        os.scandir = original_scan

    lines = json.loads(lines_path.read_text(encoding="utf-8"))
    if not isinstance(lines, dict) or not lines:
        raise ValueError("The lines file must be a nonempty mapping of names to text.")
    if any(not isinstance(s, str) or not s.strip() for s in lines.values()):
        raise ValueError("Every narration line must be nonempty text.")
    if any(Path(name).name != name for name in lines):
        raise ValueError("Narration names must be filenames, without directories.")
    selected = lines if not only else {name: lines[name] for name in only}
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    print(f"Loaded narration dependencies; making {len(selected)} lines.", flush=True)
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
    durations = read_manifest(output / "durations.json")
    provenance = read_manifest(output / "provenance.json")
    levels = provenance.get("levels", {})
    previous_lines = provenance.get("lines", {})
    speeds = provenance.get("speeds", {name: provenance.get("speed", 1.0) for name in previous_lines})
    voices = provenance.get("voices", {name: provenance.get("voice", voice) for name in previous_lines})
    timings = read_manifest(output / "token_timings.json")
    captions = read_manifest(output / "caption_phrases.json")
    for name, spoken in selected.items():
        print(f"Synthesizing {name}...", flush=True)
        chunks, tokens = [], []
        sample_offset = 0
        for result in pipe(spoken, voice=voice, speed=speed):
            chunk = np.asarray(result.audio, dtype=np.float32)
            for token in result.tokens or []:
                start, end = token.start_ts, token.end_ts
                tokens.append({
                    "text": token.text,
                    "start": None if start is None else round(sample_offset / 24000 + start, 5),
                    "end": None if end is None else round(sample_offset / 24000 + end, 5),
                })
            chunks.append(chunk)
            sample_offset += chunk.size
        if not chunks or not any(chunk.size for chunk in chunks):
            raise RuntimeError(f"No speech generated for {name}.")
        audio = np.concatenate(chunks)
        if not np.all(np.isfinite(audio)):
            raise RuntimeError(f"Non-finite speech samples in {name}.")
        peak = float(np.max(np.abs(audio)))
        if peak > 1:
            raise RuntimeError(f"Speech would clip in {name} (peak {peak:.3f}).")
        duration = round(len(audio) / 24000, 4)
        phrases = caption_phrases(name, spoken, duration, tokens)
        sf.write(output / f"{name}.wav", audio, 24000, subtype="PCM_16")
        durations[name] = duration
        levels[name] = {"peak": round(peak, 5), "rms": round(float(np.sqrt(np.mean(audio ** 2))), 5)}
        timings[name] = tokens
        captions[name] = phrases
        previous_lines[name], speeds[name], voices[name] = spoken, speed, voice
        print(f"{name}: {durations[name]:.4f} s", flush=True)
        (output / "durations.json").write_text(json.dumps(durations, indent=2) + "\n", encoding="utf-8")
    (output / "durations.json").write_text(json.dumps(durations, indent=2) + "\n", encoding="utf-8")
    # Unchanged WAVs retain their original text, levels, voice and speed.
    provenance.update({
        "model": "Kokoro-82M", "synthetic_voice": True,
        "sample_rate": 24000,
        "language": "American English", "source_method": "runs/round10/tts.py",
        "lines": previous_lines, "levels": levels, "speeds": speeds, "voices": voices,
    })
    for key, values in (("speed", speeds), ("voice", voices)):
        if len(set(values.values())) == 1:
            provenance[key] = next(iter(values.values()))
        else:
            provenance.pop(key, None)
    for name, spoken in previous_lines.items():
        if name not in captions and name in durations and name not in CAPTION_PHRASES:
            captions[name] = caption_phrases(name, spoken, durations[name], [])
    for filename, data in (("provenance.json", provenance), ("token_timings.json", timings),
                           ("caption_phrases.json", captions)):
        (output / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lines", type=Path, default=RUN / "vo_lines.json")
    ap.add_argument("--output", type=Path, default=RUN / "vo12")
    ap.add_argument("--voice", default="am_michael")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--only", nargs="+", help="Regenerate only these line names; retain other WAVs and metadata.")
    ap.add_argument("--tts-python", type=Path, default=TTS_PYTHON)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.speed <= 0:
        ap.error("--speed must be positive")
    if a.worker:
        generate(a.lines, a.output, a.voice, a.speed, a.only)
        return 0
    TEMP.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TEMP=str(TEMP), TMP=str(TEMP),
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [str(a.tts_python), "-B", "-u", str(Path(__file__).resolve()), "--worker",
               "--lines", str(a.lines.resolve()), "--output", str(a.output.resolve()),
               "--voice", a.voice, "--speed", str(a.speed)]
    if a.only:
        command.extend(["--only", *a.only])
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
