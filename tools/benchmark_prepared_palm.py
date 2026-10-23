"""Measure the real CPU palm-performance HTTP flow, including result download.

Starts and stops only its own loopback-only, --instant-only API process. Every
trial uploads audio to a fresh job; a cached video is an error. Run this alone on
the machine when comparing timings. This measures submit-to-download, not time
spent granting microphone permission or recording a voice in a browser.

Example (from the checkout, using its server-extra environment)::

    python tools/benchmark_prepared_palm.py --out runs/palm-benchmark --trials 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOICE = ROOT / "samples/voices/pf-pf-boom-author.wav"
SOURCE = ROOT / "samples/made_ahead/bending/hand_palm__boom__fire_burst"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def pipeline_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "cuesync/src/cuesync").glob("*.py"))
    paths += sorted((ROOT / "effects").glob("*.json"))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": sys.version, "logical_cpus": os.cpu_count()}
    if os.name == "nt":
        command = (
            "$p=Get-CimInstance Win32_Processor; $m=Get-CimInstance Win32_ComputerSystem; "
            "[pscustomobject]@{cpu=$p.Name;physical_cores=$p.NumberOfCores;"
            "logical_cpus=$p.NumberOfLogicalProcessors;ram_bytes=$m.TotalPhysicalMemory}|ConvertTo-Json -Compress"
        )
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True)
        if result.returncode == 0:
            info.update(json.loads(result.stdout))
    return info


def benchmark(args: argparse.Namespace) -> dict:
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"Refusing to overwrite benchmark evidence: {out}")
    voice = args.voice.resolve()
    if not voice.is_file():
        raise SystemExit(f"Voice file missing: {voice}")
    for exe in ("ffmpeg", "ffprobe"):
        if not shutil.which(exe):
            raise SystemExit(f"{exe} is required to validate the downloaded video")
    if not 1 <= args.trials <= 10:
        raise SystemExit("--trials must be between 1 and 10")
    timing = json.loads(args.timing.read_text(encoding="utf-8")) if args.timing else None
    if timing is not None and not isinstance(timing, dict):
        raise SystemExit("--timing must contain a JSON object from the voice recorder")
    sample_metadata = json.loads((ROOT / "samples/samples.json").read_text(encoding="utf-8"))
    palm = sample_metadata["plates"]["hand_palm.mp4"]
    point = palm["point"]
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", args.port))
        port = bound.getsockname()[1]
    out.mkdir(parents=True)
    start_hashes = pipeline_hashes()
    metadata = {
        "utc": datetime.now(timezone.utc).isoformat(), "hardware": hardware(),
        "command": [sys.executable, *sys.argv], "harness_sha256": digest(Path(__file__)),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "pipeline_sha256": start_hashes,
        "source_element_sha256": digest(SOURCE / "element.mp4"),
        "source_receipt_sha256": digest(SOURCE / "receipt.json"),
        "voice": {"path": str(voice), "sha256": digest(voice), "probe": probe(voice)},
        "timing": timing, "sample_metadata_sha256": digest(ROOT / "samples/samples.json"),
        "plate_sha256": digest(ROOT / "samples/plates/hand_palm.mp4"),
        "boundary": "HTTP submission start through completed final.mp4 HTTP download; no browser or recording time",
        "cold_definition": "first render after API process startup; OS file cache is not cleared",
        "cache": "all trials upload voice bytes and require cached=false; reusable element/track/matte are intentional",
        "cloud_spend_usd": 0, "gpu_available_to_server": False,
        "limitations": ["Local desktop timings, not cloud deployment performance", "No physical microphone or phone used"],
    }
    write_json(out / "metadata.json", metadata)
    command = [
        sys.executable, "-m", "cuesync.cli", "serve", "--instant-only", "--host", "127.0.0.1",
        "--port", str(port), "--jobs", str(out / "jobs"),
        "--precomputed", str(ROOT / "samples/made_ahead"),
    ]
    write_json(out / "server-command.json", command)
    results = []
    startup = time.perf_counter()
    with (out / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False) as client:
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"API exited during startup; see {out / 'server.log'}")
                    try:
                        ready = client.get("/api/health")
                        if ready.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.perf_counter() - startup > 120:
                        raise TimeoutError("API did not become ready within 120 seconds")
                    time.sleep(0.1)
                metadata["startup_seconds"] = round(time.perf_counter() - startup, 3)
                write_json(out / "health.json", ready.json())
                write_json(out / "metadata.json", metadata)
                for index in range(args.trials):
                    trial = out / f"trial-{index + 1:02d}"
                    trial.mkdir()
                    form = {"plate_sample": "hand_palm.mp4", "mode": "bending", "effect": "fire_burst",
                            "performance_mode": "reuse", "x": str(point[0]), "y": str(point[1]),
                            "frame": str(palm["hand"]["frame"])}
                    write_json(trial / "request.json", {"form": form, "timing": timing, "voice_sha256": digest(voice)})
                    tick = time.perf_counter()
                    with voice.open("rb") as audio:
                        files = {"voice": (voice.name, audio, "audio/wav")}
                        if timing is not None:
                            files["timing"] = ("timing.json", json.dumps(timing).encode(), "application/json")
                        response = client.post("/api/jobs", data=form, files=files)
                    response.raise_for_status()
                    job = response.json()
                    accepted = time.perf_counter() - tick
                    write_json(trial / "accepted.json", job)
                    if job["cached"]:
                        raise AssertionError("Benchmark received a cached video instead of a fresh performance")
                    if job["mode"] != "replay":
                        raise AssertionError(f"CPU prepared-element replay expected; got {job['mode']!r}")
                    polls = []
                    while job["status"] not in ("done", "error"):
                        polls.append({"elapsed_s": round(time.perf_counter() - tick, 3),
                                      "stage": job["stage"], "progress": job["progress"]})
                        if time.perf_counter() - tick > args.timeout:
                            raise TimeoutError(f"Job {job['id']} exceeded {args.timeout} seconds")
                        time.sleep(0.15)
                        status = client.get(f"/api/jobs/{job['id']}")
                        status.raise_for_status()
                        job = status.json()
                    finished = time.perf_counter() - tick
                    write_json(trial / "job.json", job)
                    write_json(trial / "polls.json", polls)
                    if job["status"] != "done":
                        raise RuntimeError(f"Palm performance failed: {job['error']}")
                    if "final.mp4" not in job["files"]:
                        raise AssertionError("Completed job did not advertise final.mp4; terminal response is saved in job.json")
                    download = client.get(job["files"]["final.mp4"])
                    download.raise_for_status()
                    final = trial / "palm.mp4"
                    final.write_bytes(download.content)
                    downloaded = time.perf_counter() - tick
                    events = client.get(f"/api/jobs/{job['id']}/events")
                    events.raise_for_status()
                    (trial / "events.sse").write_text(events.text, encoding="utf-8")
                    receipt_response = client.get(job["files"]["receipt.json"])
                    receipt_response.raise_for_status()
                    receipt = receipt_response.json()
                    write_json(trial / "receipt.json", receipt)
                    performed = receipt.get("bending", {}).get("performed_again", {}) or {}
                    if receipt.get("render", {}).get("seconds") != 0 or not performed.get("no_new_render"):
                        raise AssertionError("Receipt does not prove reuse with zero new model render seconds")
                    source = Path(performed["from"])
                    if digest(source / "element.mp4") != metadata["source_element_sha256"]:
                        raise AssertionError("Performance reused an unexpected source element")
                    media = probe(final)
                    decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(final), "-f", "null", "-"],
                                             capture_output=True, text=True)
                    (trial / "decode.log").write_text(decoded.stderr, encoding="utf-8")
                    if decoded.returncode or decoded.stderr.strip():
                        raise RuntimeError(f"Downloaded palm video failed full decode: {decoded.stderr}")
                    video = next(s for s in media["streams"] if s["codec_type"] == "video")
                    audio = next(s for s in media["streams"] if s["codec_type"] == "audio")
                    result = {"trial": index + 1, "process_cold": index == 0, "job_id": job["id"],
                              "accepted_seconds": round(accepted, 3), "submit_to_done_seconds": round(finished, 3),
                              "submit_to_download_seconds": round(downloaded, 3),
                              "receipt_seconds_total": receipt.get("seconds_total"),
                              "frames": int(video["nb_read_frames"]), "width": video["width"], "height": video["height"],
                              "fps": video["r_frame_rate"], "audio_sample_rate": audio["sample_rate"],
                              "final_bytes": final.stat().st_size, "final_sha256": digest(final),
                              "source_element_sha256": digest(source / "element.mp4"),
                              "final_path": str(final), "full_video_audio_decode": "passed", "cached": job["cached"],
                              "no_new_model_generation": True}
                    write_json(trial / "media.json", media)
                    write_json(trial / "result.json", result)
                    results.append(result)
                    print(json.dumps(result), flush=True)
                    if args.stop_if_over_seconds is not None and downloaded > args.stop_if_over_seconds:
                        print(f"Stopping further trials: {downloaded:.3f}s exceeds {args.stop_if_over_seconds}s", flush=True)
                        break
        except Exception as exc:
            write_json(out / "failure.json", {
                "utc": datetime.now(timezone.utc).isoformat(), "exception": type(exc).__name__,
                "message": str(exc), "completed_trials": len(results),
            })
            raise
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
            write_json(out / "server-lifecycle.json", {
                "pid": server.pid, "owned_server_stopped": server.poll() is not None,
                "process_exit_code": server.returncode, "stop_requested_by_harness": True,
            })
    end_hashes = pipeline_hashes()
    if start_hashes != end_hashes:
        write_json(out / "pipeline-after.json", end_hashes)
        raise AssertionError("Production pipeline changed during this benchmark; timings need a fresh run")
    values = [r["submit_to_download_seconds"] for r in results]
    summary = {"metadata": "metadata.json", "trials": results, "count": len(results),
               "median_submit_to_download_seconds": round(statistics.median(values), 3),
               "min_seconds": min(values), "max_seconds": max(values),
               "all_under_60_seconds": all(v < 60 for v in values), "pipeline_unchanged_during_run": True,
               "owned_server_stopped": server.poll() is not None}
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE)
    parser.add_argument("--timing", type=Path, help="Optional recorder timing JSON; omitted means zero offset")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--stop-if-over-seconds", type=float, help="Stop after a completed trial misses this target")
    args = parser.parse_args()
    print(json.dumps(benchmark(args), indent=2))


if __name__ == "__main__":
    main()
