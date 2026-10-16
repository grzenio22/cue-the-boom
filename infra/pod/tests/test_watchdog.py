"""Runs the real idle-watchdog.sh against a fake Runpod API, with seconds instead of minutes."""
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "idle-watchdog.sh"


def find_bash() -> str | None:
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if os.name == "nt":
        return str(git_bash) if git_bash.exists() else None  # avoid WSL's bash.exe
    return shutil.which("bash")


BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="needs a GNU bash")


class FakeRunpod:
    """Records requests; GraphQL answers with data (graphql_ok) or with errors."""

    def __init__(self, delete_status: int = 204, graphql_ok: bool = True):
        self.requests: list[tuple[str, str, str, str]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _record(self, status, body=b""):
                length = int(self.headers.get("Content-Length") or 0)
                sent = self.rfile.read(length).decode() if length else ""
                outer.requests.append((self.command, self.path, self.headers.get("Authorization", ""), sent))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_DELETE(self):
                self._record(delete_status)

            def do_POST(self):
                if self.path == "/graphql":
                    ok = b'{"data": {"podTerminate": null}}'
                    err = b'{"errors": [{"message": "forbidden"}], "data": null}'
                    self._record(200, ok if graphql_ok else err)
                else:
                    self._record(200)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def wait_for(self, method: str, path: str, timeout: float) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if any(r[0] == method and r[1] == path for r in self.requests):
                return True
            time.sleep(0.2)
        return False


@pytest.fixture
def run(tmp_path):
    procs = []

    def start(api, **env):
        e = {**os.environ, "RUNPOD_POD_ID": "pod123", "RUNPOD_API_KEY": "pod-scoped", "CUE_API_BASE": api.url, "CUE_GRAPHQL_URL": api.url + "/graphql",
             "CUE_STATE_DIR": (tmp_path / "state").as_posix(), "CUE_TMP": tmp_path.as_posix(), "CUE_POLL_SECONDS": "1",
             "PATH": os.environ["PATH"], **env}
        p = subprocess.Popen([BASH, str(SCRIPT)], env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
        return p

    yield start
    for p in procs:
        p.kill()


def log_text(tmp_path) -> str:
    log = tmp_path / "cue-watchdog.log"
    return log.read_text() if log.exists() else ""


def test_hard_deadline_terminates_with_the_pod_scoped_key(run, tmp_path):
    api = FakeRunpod()
    run(api, MAX_HOURS="0.0005", IDLE_MINUTES="30")  # deadline ~2 s after start
    assert api.wait_for("POST", "/graphql", 15), log_text(tmp_path)
    method, path, auth, body = api.requests[0]
    assert auth == "Bearer pod-scoped"
    assert 'podTerminate(input: {podId: \\"pod123\\"})' in body
    time.sleep(1.5)
    # Accepted, so no fallback fired (a real pod is gone by now; here the loop just repeats the call).
    assert {(m, p) for m, p, _, _ in api.requests} == {("POST", "/graphql")}
    assert "hard deadline" in log_text(tmp_path)
    assert "armed: pod pod123 key=yes" in log_text(tmp_path)  # what `podctl up` checks before READY
    assert (tmp_path / "state" / "deadline-pod123").read_text().strip().isdigit()


def test_idle_terminates_and_keepalive_postpones_it(run, tmp_path):
    api = FakeRunpod()
    run(api, MAX_HOURS="1", CUE_IDLE_SECONDS="4")
    keepalive = tmp_path / "cue-keepalive"
    end = time.time() + 6
    while time.time() < end:  # a laptop that keeps working
        keepalive.touch()
        time.sleep(0.5)
    assert not api.requests, "terminated while kept alive"
    assert api.wait_for("POST", "/graphql", 15), log_text(tmp_path)
    assert "idle for" in log_text(tmp_path)


def test_falls_back_through_every_path_when_refused(run, tmp_path):
    api = FakeRunpod(delete_status=403, graphql_ok=False)
    run(api, MAX_HOURS="0.0005")
    assert api.wait_for("POST", "/pods/pod123/action", 20), log_text(tmp_path)
    first_round = [(m, p) for m, p, _, _ in api.requests[:4]]
    assert first_round == [("POST", "/graphql"), ("DELETE", "/pods/pod123"),
                           ("POST", "/graphql"), ("POST", "/pods/pod123/action")]
    assert "podStop" in api.requests[2][3]


def test_restart_keeps_the_original_deadline(run, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "deadline-pod123").write_text(str(int(time.time()) + 2))
    api = FakeRunpod()
    run(api, MAX_HOURS="4")  # would be 4 h if the existing deadline were ignored
    assert api.wait_for("POST", "/graphql", 15), log_text(tmp_path)
