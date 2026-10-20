"""Make v11's synthetic narration with the same Kokoro-82M voice as v10.

    python tools/demo/make_v11_voice.py

The normal project Python launches the existing narration environment. Input is
runs/demo_v11/vo_lines.json ({"name": "spoken text", ...}); outputs are 24 kHz
mono WAVs and durations.json in runs/demo_v11/vo11. The default voice and speed
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
import subprocess
import sys


RUN = Path("D:/openHiggsfield/runs/demo_v11")
TTS_PYTHON = Path("D:/openHiggsfield/runs/tts-venv/Scripts/python.exe")
TEMP = Path("D:/openHiggsfield_tmp/sol6/demo/tts")


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


def generate(lines_path: Path, output: Path, voice: str, speed: float) -> None:
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
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    print(f"Loaded narration dependencies; making {len(lines)} lines.", flush=True)
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
    durations, levels = {}, {}
    for name, spoken in lines.items():
        print(f"Synthesizing {name}...", flush=True)
        chunks = [np.asarray(chunk.audio if hasattr(chunk, "audio") else chunk[2], dtype=np.float32)
                  for chunk in pipe(spoken, voice=voice, speed=speed)]
        if not chunks or not any(chunk.size for chunk in chunks):
            raise RuntimeError(f"No speech generated for {name}.")
        audio = np.concatenate(chunks)
        if not np.all(np.isfinite(audio)):
            raise RuntimeError(f"Non-finite speech samples in {name}.")
        peak = float(np.max(np.abs(audio)))
        if peak > 1:
            raise RuntimeError(f"Speech would clip in {name} (peak {peak:.3f}).")
        sf.write(output / f"{name}.wav", audio, 24000, subtype="PCM_16")
        durations[name] = round(len(audio) / 24000, 4)
        levels[name] = {"peak": round(peak, 5), "rms": round(float(np.sqrt(np.mean(audio ** 2))), 5)}
        print(f"{name}: {durations[name]:.4f} s", flush=True)
        (output / "durations.json").write_text(json.dumps(durations, indent=2) + "\n", encoding="utf-8")
    (output / "durations.json").write_text(json.dumps(durations, indent=2) + "\n", encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps({
        "model": "Kokoro-82M", "synthetic_voice": True,
        "voice": voice, "speed": speed, "sample_rate": 24000,
        "language": "American English", "source_method": "runs/round10/tts.py",
        "lines": lines, "levels": levels,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lines", type=Path, default=RUN / "vo_lines.json")
    ap.add_argument("--output", type=Path, default=RUN / "vo11")
    ap.add_argument("--voice", default="am_michael")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--tts-python", type=Path, default=TTS_PYTHON)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.speed <= 0:
        ap.error("--speed must be positive")
    if a.worker:
        generate(a.lines, a.output, a.voice, a.speed)
        return 0
    TEMP.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TEMP=str(TEMP), TMP=str(TEMP),
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [str(a.tts_python), "-B", "-u", str(Path(__file__).resolve()), "--worker",
               "--lines", str(a.lines.resolve()), "--output", str(a.output.resolve()),
               "--voice", a.voice, "--speed", str(a.speed)]
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
