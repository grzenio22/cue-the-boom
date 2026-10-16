#!/usr/bin/env python3
"""Drive the Cue the Boom GPU pod on Runpod from your laptop.

Models live on a Runpod network volume, so pods are disposable:

    podctl.py gpus                       # live stock, prices and CUDA per data center
    podctl.py volume create --dc EUR-IS-1 --size 120
    podctl.py up                         # rent a GPU pod with the volume attached
    podctl.py setup [--with-alphagen]    # install/start ComfyUI + the Comfy API proxy
    podctl.py tunnel                     # forward 127.0.0.1:8188 and :8189 (blocks)
    podctl.py ssh -- nvidia-smi          # run a command on the pod
    podctl.py down                       # terminate the pod (the volume stays)

The pod's start command arms idle-watchdog.sh before anything else, so every pod
terminates itself after --idle-minutes without work or at the --max-hours deadline,
even if this laptop goes to sleep. `up` refuses to leave a pod running unless SSH works
and the watchdog is confirmed armed.

Needs RUNPOD_API_KEY (or RUNPOD_API). `setup` also reads HF_TOKEN while weights are
still missing; it reaches the setup script through SSH stdin as an environment variable,
never a file, the volume or a command line. Uses Runpod's REST API v2 and the
standard library only.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.runpod.io/v2"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATE_DIR = REPO / ".pod"
STATE_FILE = STATE_DIR / "state.json"
LOCK_FILE = STATE_DIR / "up.lock"
DEFAULT_KEY = Path.home() / ".ssh" / "cue_pod_ed25519"

POD_NAME = "cue-the-boom"
# torch 2.12.1 built for CUDA 13.0: ComfyUI v0.41.0 only enables its fast int8 kernels on cu130+.
IMAGE = "runpod/pytorch:1.4.0-cu1300-torch2121-ubuntu2404"
MIN_CUDA = "13.0"
# Tried in order; only GPUs in the volume's data center can be placed. The 96 GB PRO 6000 keeps
# every model resident and ComfyUI's int8 kernels are tuned for it. Others need an explicit --gpu.
GPU_PREFERENCE = [
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
]
MIN_GPU_GB = 40
# Runs as the container command: arm the watchdog, then hand over to the image's /start.sh (sshd).
BOOT = ('printf %s "$CUE_WATCHDOG" | base64 -d | gunzip > /root/cue-watchdog.sh '
        '&& (setsid nohup bash /root/cue-watchdog.sh </dev/null >/dev/null 2>&1 &); exec /start.sh')


class PodError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------- API

def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API")
    if not key:
        raise PodError("set RUNPOD_API_KEY (or RUNPOD_API) to your Runpod API key")
    return key


def api(method: str, path: str, body: dict | None = None, *, timeout: float = 60):
    req = urllib.request.Request(
        API + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json",
                 "User-Agent": "cue-the-boom-podctl"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:600]
        raise PodError(f"{method} {path} -> HTTP {e.code}: {detail}", e.code) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise PodError(f"{method} {path} -> no response ({e})") from None
    return json.loads(raw) if raw.strip() else None


def get_pod(pod_id: str) -> dict | None:
    try:
        pod = api("GET", f"/pods/{pod_id}")
    except PodError as e:
        if e.status == 404:
            return None
        raise
    return None if pod.get("status") == "TERMINATED" else pod


def our_pods() -> list[dict]:
    pods, cursor = [], None
    while True:
        page = api("GET", "/pods" + (f"?cursor={cursor}" if cursor else ""))
        pods += [p for p in page.get("pods", []) if (p.get("name") or "").startswith(POD_NAME)
                 and p.get("status") != "TERMINATED"]
        pagination = page.get("pagination") or {}
        if not pagination.get("hasNextPage"):
            return pods
        cursor = pagination["nextCursor"]


def terminate(pod_id: str) -> bool:
    """Delete a pod; True once it is gone (already-gone counts)."""
    try:
        api("DELETE", f"/pods/{pod_id}")
    except PodError as e:
        if e.status != 404:
            raise
    return get_pod(pod_id) is None


# ----------------------------------------------------------------------- helpers

def ssh_endpoint(pod: dict) -> tuple[str, int] | None:
    """Host and port of the pod's direct (exposed TCP) SSH, once Runpod reports it."""
    direct = (pod.get("ssh") or {}).get("direct") or {}
    host, port = direct.get("host"), direct.get("port")
    return (host, int(port)) if host and port else None


def watchdog_blob() -> str:
    script = (HERE / "idle-watchdog.sh").read_bytes().replace(b"\r\n", b"\n")
    return base64.b64encode(gzip.compress(script, mtime=0)).decode()


def create_body(*, name: str, gpu: str, volume_id: str, data_center: str, image: str, public_key: str,
                idle_minutes: int, max_hours: float, min_ram_gb: int = 96, container_disk_gb: int = 40) -> dict:
    """POST /v2/pods body: a Secure Cloud GPU pod, SSH as the only exposed port, watchdog armed at boot."""
    return {
        "name": name,
        "image": image,
        "cloud": "SECURE",
        "gpu": {"id": gpu, "count": 1, "minCudaVersion": MIN_CUDA, "minRamPerGpu": min_ram_gb},
        "dataCenterIds": [data_center],
        "disk": container_disk_gb,
        "ports": ["22/tcp"],
        "mounts": {"network": [{"volumeId": volume_id, "path": "/workspace"}]},
        "cmd": ["bash", "-c", BOOT],
        "env": {
            "PUBLIC_KEY": public_key,
            "IDLE_MINUTES": str(idle_minutes),
            "MAX_HOURS": str(max_hours),
            "CUE_WATCHDOG": watchdog_blob(),
        },
    }


def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def volume_only(state: dict) -> dict:
    return {k: v for k, v in state.items() if k in ("volume_id", "data_center")}


def require_pod(state: dict) -> dict:
    if not state.get("pod_id") or not state.get("host"):
        raise PodError("no pod is up; run `podctl.py up` first")
    return state


def public_key_path(key: Path) -> Path:
    return Path(str(key) + ".pub")


def ensure_key(key: Path) -> str:
    if not key.exists():
        key.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", POD_NAME, "-f", str(key)], check=True)
    return public_key_path(key).read_text().strip()


def known_hosts(state: dict) -> Path:
    return STATE_DIR / f"known_hosts-{state['pod_id']}"


def ssh_opts(state: dict, *, batch: bool = True) -> list[str]:
    opts = [
        "-i", state["key"],
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f'UserKnownHostsFile="{known_hosts(state).as_posix()}"',
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=4",
        "-o", "ConnectTimeout=15",
        "-o", "LogLevel=ERROR",
    ]
    return opts + (["-o", "BatchMode=yes"] if batch else [])


def ssh_cmd(state: dict, remote: str, *, tty: bool = False) -> list[str]:
    return ["ssh", *(["-t"] if tty else []), *ssh_opts(state, batch=not tty),
            "-p", str(state["port"]), f"root@{state['host']}", remote]


def run_ssh(state: dict, remote: str, *, input: bytes | None = None, check: bool = True,
            capture: bool = False, timeout: float | None = None,
            keepalive: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command on the pod. By default it counts as activity for the idle watchdog."""
    if keepalive:
        remote = f"touch /tmp/cue-keepalive 2>/dev/null; {remote}"
    return subprocess.run(ssh_cmd(state, remote), input=input, check=check, timeout=timeout,
                          stdout=subprocess.PIPE if capture else None)


def probe(state: dict, remote: str, timeout: float = 30) -> str | None:
    """Output of a short command, or None if it fails or times out."""
    try:
        out = run_ssh(state, remote, check=False, capture=True, timeout=timeout, keepalive=False)
    except subprocess.TimeoutExpired:
        return None
    return out.stdout.decode(errors="replace").strip() if out.returncode == 0 else None


def scp(state: dict, src: str, dst: str) -> None:
    run_ssh(state, "true", timeout=60)  # refresh the keepalive before a possibly long transfer
    subprocess.run(["scp", *ssh_opts(state), "-P", str(state["port"]), src, dst], check=True)


def remote_path(state: dict, path: str) -> str:
    return f"root@{state['host']}:{path}"


class UpLock:
    """Stops two `up` runs from racing each other into two billed pods."""

    def __enter__(self):
        STATE_DIR.mkdir(exist_ok=True)
        if LOCK_FILE.exists() and time.time() - LOCK_FILE.stat().st_mtime > 3600:
            LOCK_FILE.unlink()  # stale: a crashed run
        try:
            os.close(os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            raise PodError(f"another `up` is running (delete {LOCK_FILE} if it isn't)") from None
        return self

    def __exit__(self, *exc):
        LOCK_FILE.unlink(missing_ok=True)


def reap_by_name(name: str, attempts: int = 6) -> None:
    """Terminate any pod with this exact name. Covers a create whose response was lost."""
    for attempt in range(attempts):
        found = [p for p in our_pods() if p.get("name") == name]
        for p in found:
            print(f"terminating {p['id']} ({name})", file=sys.stderr)
            terminate(p["id"])
        if found or attempt == attempts - 1:
            return
        time.sleep(10)


# ---------------------------------------------------------------------- commands

def cmd_gpus(args) -> None:
    dcs = api("GET", "/catalog/datacenters?include=GPU_AVAILABILITY&networkVolumeTypes=STANDARD")["dataCenters"]
    catalog = {g["id"]: g for g in api("GET", "/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE")["gpus"]}
    print(f"{'data center':<11} {'GPU':<52} {'$/h':>6}  {'stock':<6} CUDA")
    for dc in sorted(dcs, key=lambda d: d["id"]):
        if args.dc and dc["id"] != args.dc:
            continue
        for g in dc.get("gpuAvailability") or []:
            info = catalog.get(g["id"]) or {}
            if g.get("availability") in (None, "NONE") or (info.get("memory") or 0) < MIN_GPU_GB:
                continue
            cuda = ",".join(c["version"] for c in info.get("cudaVersions") or [] if c.get("available"))
            price = (info.get("price") or {}).get("secure") or 0
            print(f"{dc['id']:<11} {g['id']:<52} {price:>6.2f}  {g['availability']:<6} {cuda}")


def cmd_volume(args) -> None:
    state = load_state()
    if args.action == "list":
        for v in api("GET", "/network-volumes").get("networkVolumes", []):
            mark = "  <- in use" if v["id"] == state.get("volume_id") else ""
            print(f"{v['id']}  {v['name']}  {v['size']} GB  {v['dataCenter']}  {v.get('type')}{mark}")
        return
    if state.get("volume_id"):
        raise PodError(f"a volume is already configured ({state['volume_id']}); see `volume list`")
    vol = api("POST", "/network-volumes", {"name": args.name, "size": args.size,
                                           "dataCenter": args.dc, "type": "STANDARD"})
    state.update(volume_id=vol["id"], data_center=vol["dataCenter"])
    save_state(state)
    print(f"created network volume {vol['id']} ({vol['size']} GB in {vol['dataCenter']}, ~${vol['size'] * 0.07:.2f}/month)")


def cmd_up(args) -> None:
    if not (5 <= args.idle_minutes <= 240):
        raise PodError("--idle-minutes must be between 5 and 240")
    if not (math.isfinite(args.max_hours) and 0.25 <= args.max_hours <= 24):
        raise PodError("--max-hours must be between 0.25 and 24")
    state = load_state()
    if not state.get("volume_id"):
        raise PodError("no network volume yet; run `podctl.py volume create --dc <DC>` first")
    key = Path(args.key).expanduser()
    gpus = args.gpu or GPU_PREFERENCE
    name = f"{POD_NAME}-{secrets.token_hex(3)}"
    common = dict(name=name, volume_id=state["volume_id"], data_center=state["data_center"], image=args.image,
                  idle_minutes=args.idle_minutes, max_hours=args.max_hours, min_ram_gb=args.min_ram)
    if args.dry_run:
        body = create_body(gpu=gpus[0], public_key="<public key>", **common)
        body["env"]["CUE_WATCHDOG"] = f"<{len(body['env']['CUE_WATCHDOG'])} chars>"
        print(json.dumps(body, indent=2))
        return

    with UpLock():
        existing = our_pods()
        if existing:
            raise PodError("pods already exist: " + ", ".join(f"{p['id']} ({p['status']})" for p in existing)
                           + "; run `podctl.py down --all` first")
        public_key = ensure_key(key)
        state.update(pending=name, key=str(key))
        save_state(state)
        try:
            pod = None
            for gpu in gpus:
                try:
                    pod = api("POST", "/pods", create_body(gpu=gpu, public_key=public_key, **common))
                    break
                except PodError as e:
                    if e.status not in (400, 409, 503):
                        raise
                    print(f"  {gpu}: not available ({str(e)[-160:]})")
            if pod is None:
                raise PodError(f"none of {len(gpus)} GPU types could be placed in {state['data_center']}; "
                               "check `podctl.py gpus`")
            state.update(pod_id=pod["id"], gpu=(pod.get("gpu") or {}).get("id", gpu), created=time.time(),
                         cost_per_hr=float(pod.get("cost") or 0), idle_minutes=args.idle_minutes,
                         host=None, port=None)
            state.pop("pending", None)
            save_state(state)
            if state["cost_per_hr"] > args.max_price:
                raise PodError(f"{state['gpu']} costs ${state['cost_per_hr']:.2f}/h, above --max-price {args.max_price:.2f}")
            print(f"pod {pod['id']} ({state['gpu']}) created at ${state['cost_per_hr']:.2f}/h; waiting for SSH...",
                  flush=True)
            wait_for_ssh(state, args.boot_timeout)
            verify_watchdog(state)
        except BaseException:
            print("terminating the pod that failed to come up", file=sys.stderr)
            try:
                if state.get("pod_id"):
                    terminate(state["pod_id"])
                reap_by_name(name)
            except Exception as cleanup_error:  # keep the original error; make the leftover loud
                print(f"CLEANUP FAILED ({cleanup_error}); run `podctl.py down --all`", file=sys.stderr)
            else:
                save_state(volume_only(state))
            raise
    gpu_line = probe(state, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader") or "?"
    print(f"READY  {gpu_line}  ssh -p {state['port']} root@{state['host']}")


def wait_for_ssh(state: dict, boot_timeout: float) -> None:
    deadline = time.time() + boot_timeout
    blips = 0
    while time.time() < deadline:
        try:
            pod = get_pod(state["pod_id"])
            blips = 0
        except PodError as e:  # a network or server blip is not a reason to throw the pod away
            blips += 1
            if (e.status is not None and e.status < 500) or blips > 5:
                raise
            time.sleep(10)
            continue
        if pod is None or pod.get("status") in ("ERROR", "EXITED"):
            raise PodError(f"pod {state['pod_id']} is {pod and pod.get('status') or 'gone'}")
        endpoint = ssh_endpoint(pod)
        if endpoint:
            state["host"], state["port"] = endpoint
            save_state(state)
            if probe(state, "true", timeout=min(30, max(5, deadline - time.time()))) is not None:
                return
        time.sleep(10)
    raise PodError(f"pod did not accept SSH within {boot_timeout:.0f} s")


def verify_watchdog(state: dict) -> None:
    """The watchdog must be running, hold this pod's deadline and have a key to terminate with."""
    check = (f"pgrep -f cue-watchdog.sh >/dev/null && grep -q 'armed: pod {state['pod_id']} key=yes' /tmp/cue-watchdog.log "
             f"&& d=$(cat /workspace/.cue/deadline-{state['pod_id']}) && echo $(( (d - $(date +%s)) / 60 ))")
    for _ in range(6):
        minutes = probe(state, check)
        if minutes and minutes.lstrip("-").isdigit():
            print(f"watchdog armed: terminates after {state.get('idle_minutes', '?')} idle minutes "
                  f"or in {int(minutes)} min at the latest")
            return
        time.sleep(10)
    log = probe(state, "tail -n 5 /tmp/cue-watchdog.log") or "no watchdog log"
    raise PodError(f"the idle watchdog is not armed on the pod ({log})")


def cmd_status(args) -> None:
    state = load_state()
    pods = our_pods()
    if not pods:
        where = f" (volume {state['volume_id']} in {state.get('data_center')})" if state.get("volume_id") else ""
        print("no pods running" + where)
    for p in pods:
        tracked = p["id"] == state.get("pod_id")
        uptime_h = ((p.get("runtime") or {}).get("uptime") or 0) / 3600
        cost = float(p.get("cost") or 0)
        util = ",".join(str(g.get("util", g.get("gpuUtilPercent", "?"))) for g in (p.get("runtime") or {}).get("gpus") or [])
        print(f"{p['id']}  {p.get('name')}  {p.get('status')}  ${cost:.2f}/h  up {uptime_h:.2f} h  "
              f"gpu util {util or '-'}%{'' if tracked else '  (NOT tracked by this laptop)'}")
        if tracked and p.get("status") == "RUNNING" and state.get("host"):
            info = probe(state, f"tail -n 2 /tmp/cue-watchdog.log; d=$(cat /workspace/.cue/deadline-{p['id']}); "
                                'echo "deadline in $(( (d - $(date +%s)) / 60 )) min, '
                                'idle $(( ($(date +%s) - $(stat -c %Y /tmp/cue-keepalive)) / 60 )) min"')
            print(info or "  (watchdog state unavailable)")


def cmd_ssh(args) -> None:
    state = require_pod(load_state())
    if not args.command:
        sys.exit(subprocess.run(ssh_cmd(state, "touch /tmp/cue-keepalive; exec bash -l", tty=True)).returncode)
    sys.exit(run_ssh(state, " ".join(args.command), check=False).returncode)


def cmd_push(args) -> None:
    state = require_pod(load_state())
    for f in args.files:
        scp(state, f, remote_path(state, args.dest))


def cmd_pull(args) -> None:
    state = require_pod(load_state())
    scp(state, remote_path(state, args.src), args.dest)


def cmd_setup(args) -> None:
    state = require_pod(load_state())
    scp(state, str(HERE / "pod-setup.sh"), remote_path(state, "/workspace/pod-setup.sh"))
    token = (os.environ.get("HF_TOKEN") or "").strip()
    flags = " --with-alphagen" if args.with_alphagen else ""
    remote = ('IFS= read -r HF_TOKEN; if [ -n "$HF_TOKEN" ]; then export HF_TOKEN; else unset HF_TOKEN; fi; '
              f"sed -i 's/\\r$//' /workspace/pod-setup.sh && bash /workspace/pod-setup.sh{flags}")
    sys.exit(run_ssh(state, remote, input=(token + "\n").encode(), check=False).returncode)


SYNC_PATHS = ["cuesync/src", "cuesync/pyproject.toml", "cuesync/README.md", "comfyui-cuesync", "effects", "workflows",
              "infra/pod/pod-setup.sh"]


def cmd_sync(args) -> None:
    """Ship this checkout's cuesync engine, the CueSync node pack, effects and workflows to the pod, install
    cuesync into the pod's venv and link the node pack into ComfyUI (restart ComfyUI to load new nodes)."""
    import io
    import tarfile

    state = require_pod(load_state())
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in SYNC_PATHS:
            if (REPO / rel).exists():
                tar.add(REPO / rel, arcname=rel,
                        filter=lambda t: None if "__pycache__" in t.name or t.name.endswith(".pyc") else t)
    remote = ("rm -rf /workspace/cue-src && mkdir -p /workspace/cue-src && tar -xzf - -C /workspace/cue-src && "
              "/workspace/venv/bin/python -m pip install -q --no-deps -e /workspace/cue-src/cuesync && "
              "ln -sfn /workspace/cue-src/comfyui-cuesync /workspace/ComfyUI/custom_nodes/comfyui-cuesync && "
              "cp /workspace/cue-src/infra/pod/pod-setup.sh /workspace/pod-setup.sh && "
              "echo synced $(du -sh /workspace/cue-src | cut -f1)")
    run_ssh(state, remote, input=buf.getvalue(), timeout=600)
    if args.restart:
        # [.] / [y]: the patterns must not match this very command line, or pkill kills its own shell
        run_ssh(state, "pkill -f 'main[.]py --listen 127.0.0.1 --port 8188'; pkill -f 'comfy-api-prox[y]'; sleep 3; "
                       "bash /workspace/pod-setup.sh > /workspace/logs/restart.log 2>&1; tail -n 3 /workspace/logs/restart.log",
                timeout=900)


def cmd_tunnel(args) -> None:
    state = require_pod(load_state())
    run_ssh(state, "true", timeout=60)
    print("tunnel up: ComfyUI http://127.0.0.1:8188  Comfy API http://127.0.0.1:8189  (Ctrl+C closes)", flush=True)
    # An open tunnel is a working session (e.g. editing a graph in the browser): keep the pod
    # alive while it lasts. The hard deadline still bounds a forgotten tunnel.
    sys.exit(subprocess.run(["ssh", *ssh_opts(state), "-o", "ExitOnForwardFailure=yes",
                             "-L", "127.0.0.1:8188:127.0.0.1:8188", "-L", "127.0.0.1:8189:127.0.0.1:8189",
                             "-p", str(state["port"]), f"root@{state['host']}",
                             "while true; do touch /tmp/cue-keepalive; sleep 60; done"]).returncode)


def cmd_extend(args) -> None:
    state = require_pod(load_state())
    if not (math.isfinite(args.hours) and 0 <= args.hours <= 24):
        raise PodError("hours must be between 0 and 24")
    f = f"/workspace/.cue/deadline-{state['pod_id']}"
    run_ssh(state, f"echo $(( $(date +%s) + {int(args.hours * 3600)} )) > {f} "
                   f"&& echo deadline moved to $(date -u -d @$(cat {f}) +%H:%MZ)", timeout=60)


def cmd_down(args) -> None:
    state = load_state()
    ids = [state["pod_id"]] if state.get("pod_id") else []
    others = [p["id"] for p in our_pods() if p["id"] not in ids]
    if args.all:
        ids += others
    elif others:
        print(f"note: untracked pods still running: {', '.join(others)} (use --all)", file=sys.stderr)
    if not ids:
        print("nothing to terminate")
    failed = []
    for pod_id in ids:
        try:
            if terminate(pod_id):
                print(f"terminated {pod_id}")
            else:
                failed.append(f"{pod_id} still exists")
        except PodError as e:
            failed.append(f"{pod_id}: {e}")
    if state.get("pod_id") and not any(f.startswith(state["pod_id"]) for f in failed):
        hours = (time.time() - state.get("created", time.time())) / 3600
        print(f"session: {hours:.2f} h, ~${hours * float(state.get('cost_per_hr') or 0):.2f}")
        (STATE_DIR / f"known_hosts-{state['pod_id']}").unlink(missing_ok=True)
        save_state(volume_only(state))
    if failed:
        raise PodError("could not terminate: " + "; ".join(failed) + " (retry, or use the Runpod console)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="podctl.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("gpus", help="live GPU stock, Secure Cloud prices and CUDA (volume data centers, 40 GB+)")
    s.add_argument("--dc", help="only this data center")
    s.set_defaults(fn=cmd_gpus)

    s = sub.add_parser("volume", help="create or list the network volume that holds the models")
    s.add_argument("action", choices=["create", "list"])
    s.add_argument("--dc", default="EUR-IS-1")
    s.add_argument("--size", type=int, default=120, help="GB (can grow later, never shrink)")
    s.add_argument("--name", default=f"{POD_NAME}-models")
    s.set_defaults(fn=cmd_volume)

    s = sub.add_parser("up", help="rent a GPU pod with the volume attached")
    s.add_argument("--gpu", action="append", help="GPU type id (repeatable, in preference order)")
    s.add_argument("--image", default=IMAGE)
    s.add_argument("--key", default=str(DEFAULT_KEY), help="SSH private key (created if missing)")
    s.add_argument("--idle-minutes", type=int, default=30)
    s.add_argument("--max-hours", type=float, default=4)
    s.add_argument("--max-price", type=float, default=3.0, help="terminate at once if the pod costs more ($/h)")
    s.add_argument("--min-ram", type=int, default=96, help="minimum host RAM in GB (placement filter)")
    s.add_argument("--boot-timeout", type=float, default=900, help="seconds to wait for SSH")
    s.add_argument("--dry-run", action="store_true", help="print the request body and exit")
    s.set_defaults(fn=cmd_up)

    s = sub.add_parser("status", help="pods, cost, watchdog state (does not count as activity)")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("setup", help="install/start ComfyUI + Comfy API proxy on the pod")
    s.add_argument("--with-alphagen", action="store_true")
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("sync", help="ship cuesync + the CueSync node pack to the pod and install them")
    s.add_argument("--restart", action="store_true", help="restart ComfyUI and the proxy afterwards (loads new nodes)")
    s.set_defaults(fn=cmd_sync)

    s = sub.add_parser("tunnel", help="forward 127.0.0.1:8188 and :8189 to the pod (blocks)")
    s.set_defaults(fn=cmd_tunnel)

    s = sub.add_parser("ssh", help="run a command on the pod (or open a shell)")
    s.add_argument("command", nargs=argparse.REMAINDER, help="joined with spaces and run by the remote shell")
    s.set_defaults(fn=cmd_ssh)

    s = sub.add_parser("push", help="copy local files to the pod")
    s.add_argument("files", nargs="+")
    s.add_argument("dest")
    s.set_defaults(fn=cmd_push)

    s = sub.add_parser("pull", help="copy a file from the pod")
    s.add_argument("src")
    s.add_argument("dest")
    s.set_defaults(fn=cmd_pull)

    s = sub.add_parser("extend", help="move the hard deadline to N hours from now (0 = terminate within a minute)")
    s.add_argument("hours", type=float)
    s.set_defaults(fn=cmd_extend)

    s = sub.add_parser("down", help="terminate the pod (the volume and models stay)")
    s.add_argument("--all", action="store_true", help=f"also terminate untracked pods named {POD_NAME}*")
    s.set_defaults(fn=cmd_down)

    args = p.parse_args(argv)
    if getattr(args, "command", None) and args.command[:1] == ["--"]:
        args.command = args.command[1:]
    try:
        args.fn(args)
    except PodError as e:
        sys.exit(f"podctl: {e}")


if __name__ == "__main__":
    main()
