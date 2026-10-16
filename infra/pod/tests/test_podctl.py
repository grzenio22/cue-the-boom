import base64
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("podctl", Path(__file__).resolve().parents[1] / "podctl.py")
podctl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(podctl)


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(podctl, "STATE_DIR", tmp_path)
    monkeypatch.setattr(podctl, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(podctl, "LOCK_FILE", tmp_path / "up.lock")
    return tmp_path


def body(**overrides):
    args = dict(name="cue-the-boom-abc123", gpu="NVIDIA RTX PRO 6000 Blackwell Server Edition", volume_id="vol123",
                data_center="EUR-IS-1", image="img:tag", public_key="ssh-ed25519 AAAA test",
                idle_minutes=30, max_hours=4.0)
    return podctl.create_body(**{**args, **overrides})


def test_ssh_endpoint_waits_for_the_direct_endpoint():
    assert podctl.ssh_endpoint({}) is None
    assert podctl.ssh_endpoint({"ssh": {"proxy": {"host": "ssh.runpod.io", "port": 22}, "direct": None}}) is None
    assert podctl.ssh_endpoint({"ssh": {"direct": {"host": "", "port": 40100}}}) is None
    pod = {"ssh": {"direct": {"host": "203.0.113.7", "port": "40100", "username": "root"}}}
    assert podctl.ssh_endpoint(pod) == ("203.0.113.7", 40100)


def test_create_body_exposes_only_ssh_and_mounts_the_volume():
    b = body()
    assert b["ports"] == ["22/tcp"]
    assert b["mounts"] == {"network": [{"volumeId": "vol123", "path": "/workspace"}]}
    assert b["dataCenterIds"] == ["EUR-IS-1"]
    assert b["cloud"] == "SECURE"
    assert b["gpu"]["id"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert b["gpu"]["minCudaVersion"] == "13.0"
    assert b["env"]["PUBLIC_KEY"] == "ssh-ed25519 AAAA test"
    assert "HF_TOKEN" not in b["env"]
    json.dumps(b)


def test_boot_command_arms_the_watchdog_then_runs_the_image_start_script():
    b = body(idle_minutes=20, max_hours=2.5)
    assert b["cmd"][:2] == ["bash", "-c"]
    assert b["cmd"][2].endswith("exec /start.sh")
    assert b["env"]["IDLE_MINUTES"] == "20"
    assert b["env"]["MAX_HOURS"] == "2.5"
    script = gzip.decompress(base64.b64decode(b["env"]["CUE_WATCHDOG"]))
    assert script == (Path(podctl.__file__).parent / "idle-watchdog.sh").read_bytes()


def test_api_key_accepts_either_variable(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_API", raising=False)
    with pytest.raises(podctl.PodError):
        podctl.api_key()
    monkeypatch.setenv("RUNPOD_API", "test-key-x")
    assert podctl.api_key() == "test-key-x"
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key-y")
    assert podctl.api_key() == "test-key-y"


def test_public_key_path_keeps_the_full_name():
    assert podctl.public_key_path(Path("keys/identity.pem")) == Path("keys/identity.pem.pub")


def test_up_dry_run_prints_body_without_key_or_blob(state_dir, capsys):
    podctl.save_state({"volume_id": "vol123", "data_center": "EUR-IS-1"})
    podctl.main(["up", "--dry-run"])
    out = capsys.readouterr().out
    assert '"volumeId": "vol123"' in out
    assert "<public key>" in out
    assert "H4sI" not in out  # the gzip+base64 watchdog blob is summarised, not dumped


@pytest.mark.parametrize("argv, message", [
    (["up", "--max-hours", "nan"], "--max-hours"),
    (["up", "--max-hours", "100"], "--max-hours"),
    (["up", "--idle-minutes", "1"], "--idle-minutes"),
])
def test_up_validates_limits_before_renting(state_dir, argv, message):
    podctl.save_state({"volume_id": "vol123", "data_center": "EUR-IS-1"})
    with pytest.raises(SystemExit, match=message):
        podctl.main(argv)


def test_up_refuses_without_a_volume(state_dir):
    with pytest.raises(SystemExit, match="no network volume"):
        podctl.main(["up", "--dry-run"])


def test_up_refuses_when_pods_already_exist(state_dir, monkeypatch):
    podctl.save_state({"volume_id": "vol123", "data_center": "EUR-IS-1"})
    monkeypatch.setattr(podctl, "our_pods", lambda: [{"id": "p1", "status": "RUNNING"}])
    monkeypatch.setattr(podctl, "api", lambda *a, **k: pytest.fail("must not create a pod"))
    with pytest.raises(SystemExit, match="already exist"):
        podctl.main(["up"])
    assert not (state_dir / "up.lock").exists()


def test_up_terminates_the_pod_when_boot_fails(state_dir, monkeypatch, tmp_path):
    podctl.save_state({"volume_id": "vol123", "data_center": "EUR-IS-1"})
    key = tmp_path / "id_test"
    key.write_text("private")
    Path(str(key) + ".pub").write_text("ssh-ed25519 AAAA\n")
    calls = []

    def fake_api(method, path, body=None, **kw):
        calls.append((method, path))
        if (method, path) == ("POST", "/pods"):
            return {"id": "pod1", "cost": 2.49, "gpu": {"id": body["gpu"]["id"]}}
        if method == "GET" and path == "/pods":
            return {"pods": [], "pagination": {"hasNextPage": False}}
        if (method, path) == ("DELETE", "/pods/pod1"):
            return None
        if (method, path) == ("GET", "/pods/pod1"):
            if ("DELETE", "/pods/pod1") in calls:
                raise podctl.PodError("gone", 404)
            return {"id": "pod1", "status": "RUNNING"}
        raise AssertionError((method, path))

    monkeypatch.setattr(podctl, "api", fake_api)
    monkeypatch.setattr(podctl, "wait_for_ssh", lambda *a: (_ for _ in ()).throw(podctl.PodError("no ssh")))
    monkeypatch.setattr(podctl.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match="no ssh"):
        podctl.main(["up", "--key", str(key)])
    assert ("DELETE", "/pods/pod1") in calls
    assert podctl.load_state() == {"volume_id": "vol123", "data_center": "EUR-IS-1"}


def test_up_tries_the_next_gpu_when_one_cannot_be_placed(state_dir, monkeypatch, tmp_path):
    podctl.save_state({"volume_id": "vol123", "data_center": "EUR-IS-1"})
    key = tmp_path / "id_test"
    key.write_text("private")
    Path(str(key) + ".pub").write_text("ssh-ed25519 AAAA\n")
    tried = []

    def fake_api(method, path, body=None, **kw):
        if method == "GET" and path == "/pods":
            return {"pods": [], "pagination": {"hasNextPage": False}}
        if (method, path) == ("POST", "/pods"):
            tried.append(body["gpu"]["id"])
            if len(tried) == 1:
                raise podctl.PodError("no capacity", 400)
            return {"id": "pod2", "cost": "2.59"}
        raise AssertionError((method, path))

    monkeypatch.setattr(podctl, "api", fake_api)
    monkeypatch.setattr(podctl, "wait_for_ssh", lambda state, t: state.update(host="h", port=1))
    monkeypatch.setattr(podctl, "verify_watchdog", lambda state: None)
    monkeypatch.setattr(podctl, "probe", lambda *a, **k: "RTX PRO 6000, 97887 MiB")
    podctl.main(["up", "--key", str(key), "--gpu", "A", "--gpu", "B"])
    assert tried == ["A", "B"]
    s = podctl.load_state()
    assert s["pod_id"] == "pod2" and s["cost_per_hr"] == 2.59 and "pending" not in s


def test_down_attempts_every_pod_and_reports_failures(state_dir, monkeypatch):
    podctl.save_state({"volume_id": "v", "data_center": "EUR-IS-1", "pod_id": "a", "created": 0, "cost_per_hr": "2.49"})
    monkeypatch.setattr(podctl, "our_pods", lambda: [{"id": "a"}, {"id": "b"}, {"id": "c"}])
    attempted = []

    def fake_terminate(pod_id):
        attempted.append(pod_id)
        if pod_id == "b":
            raise podctl.PodError("boom", 500)
        return True

    monkeypatch.setattr(podctl, "terminate", fake_terminate)
    with pytest.raises(SystemExit, match="b: boom"):
        podctl.main(["down", "--all"])
    assert attempted == ["a", "b", "c"]
    assert podctl.load_state() == {"volume_id": "v", "data_center": "EUR-IS-1"}


def test_up_terminates_a_pod_above_max_price(state_dir, monkeypatch, tmp_path):
    podctl.save_state({"volume_id": "vol123", "data_center": "EUR-IS-1"})
    key = tmp_path / "id_test"
    key.write_text("private")
    Path(str(key) + ".pub").write_text("ssh-ed25519 AAAA\n")
    deleted = []

    def fake_api(method, path, body=None, **kw):
        if method == "GET" and path == "/pods":
            return {"pods": [], "pagination": {"hasNextPage": False}}
        if (method, path) == ("POST", "/pods"):
            return {"id": "pricey", "cost": 3.99}
        if method == "DELETE":
            deleted.append(path)
            return None
        if method == "GET":
            raise podctl.PodError("gone", 404)
        raise AssertionError((method, path))

    monkeypatch.setattr(podctl, "api", fake_api)
    monkeypatch.setattr(podctl, "wait_for_ssh", lambda *a: pytest.fail("must not wait for an over-budget pod"))
    monkeypatch.setattr(podctl.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match="max-price"):
        podctl.main(["up", "--key", str(key), "--gpu", "X"])
    assert deleted == ["/pods/pricey"]


def test_wait_for_ssh_rides_out_network_blips(state_dir, monkeypatch):
    calls = {"n": 0}

    def flaky(pod_id):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise podctl.PodError("no response")
        return {"status": "RUNNING", "ssh": {"direct": {"host": "h", "port": 2222}}}

    monkeypatch.setattr(podctl, "get_pod", flaky)
    monkeypatch.setattr(podctl, "probe", lambda *a, **k: "")
    monkeypatch.setattr(podctl.time, "sleep", lambda s: None)
    state = {"pod_id": "p", "key": "k"}
    podctl.wait_for_ssh(state, 600)
    assert (state["host"], state["port"]) == ("h", 2222)
