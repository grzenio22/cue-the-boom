"""Readiness must describe the workflow's resources, not just a live proxy."""

import json
import http.client
import urllib.error
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cuesync.pipeline import ComfyRenderer


@pytest.fixture
def backend(monkeypatch, tmp_path):
    for name in ("COMFY_BASE_URL", "COMFY_API_KEY", "COMFYUI_BASE_URL", "COMFYUI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COMFY_BASE_URL", "http://127.0.0.1:8189")
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps({
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "fire.safetensors"}},
        "2": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "result"}},
        "3": {"class_type": "LTXVImgToVideoInplace", "inputs": {}},
        "4": {"class_type": "LoadVideo", "inputs": {"file": "guide.mp4"}},
        "5": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "ainvfxfluid, fire"}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
    }))
    return ComfyRenderer(workflow)


def node_info(*, models=("fire.safetensors",), scene=True):
    result = {
        "UNETLoader": {"input": {"required": {"unet_name": [list(models), {}]}}},
        "SaveVideo": {"input": {"required": {"filename_prefix": ["STRING", {}]}}},
        "LTXVImgToVideoInplace": {"input": {"required": {}}},
        "LoadVideo": {"input": {"required": {"file": [[], {}]}}},
        "PrimitiveStringMultiline": {"input": {"required": {"value": ["STRING", {}]}}},
        "RandomNoise": {"input": {"required": {"noise_seed": ["INT", {}]}}},
    }
    if scene:
        result["LoadImage"] = {"input": {"required": {"image": [[], {}]}}}
    return result


def install_responses(monkeypatch, info, *, base="http://127.0.0.1:8189",
                      inventory="http://127.0.0.1:8188/object_info"):
    """Only the network boundary is fake; inputs and status computation are real."""
    from cuesync import readiness

    calls = []

    def fetch(url, headers):
        calls.append((url, headers))
        if url == base + "/api/v2/jobs?limit=1":
            return {"jobs": []}
        if url == inventory:
            return info
        raise AssertionError(f"unexpected endpoint {url}")

    monkeypatch.setattr(readiness, "_fetch_json", fetch)
    return calls


def test_live_proxy_with_missing_weights_is_not_render_ready(backend, monkeypatch):
    install_responses(monkeypatch, node_info(models=()))
    status = backend.preflight()
    assert status.connected and not status.ready
    assert status.status == "missing_resources"
    assert status.missing_models == ("UNETLoader.unet_name: fire.safetensors",)
    assert "fire.safetensors" in status.message


def test_missing_node_is_reported_without_irrelevant_optional_nodes(backend, monkeypatch):
    info = node_info()
    del info["SaveVideo"]
    install_responses(monkeypatch, info)
    status = backend.preflight()
    assert status.missing_nodes == ("SaveVideo",)
    assert "CueSyncObjectMatte" not in status.missing_nodes
    assert not status.ready


def test_ready_when_all_workflow_resources_are_present(backend, monkeypatch):
    install_responses(monkeypatch, node_info())
    status = backend.preflight()
    assert status.connected and status.ready and status.scene_ready
    assert backend.ready()


def test_missing_scene_loader_does_not_disable_element_render(backend, monkeypatch):
    install_responses(monkeypatch, node_info(scene=False))
    status = backend.preflight()
    assert status.ready and not status.scene_ready
    assert "LoadImage" in status.message


def test_no_url_or_cloud_key_is_not_assumed_ready(backend, monkeypatch):
    monkeypatch.delenv("COMFY_BASE_URL")
    assert not backend.ready()
    assert backend.preflight().status == "not_configured"
    assert "COMFY_API_KEY" in backend.preflight().message


def test_key_only_configuration_uses_documented_cloud_inventory(backend, monkeypatch):
    monkeypatch.delenv("COMFY_BASE_URL")
    monkeypatch.setenv("COMFY_API_KEY", "test-key")
    calls = install_responses(monkeypatch, node_info(), base="https://cloud.comfy.org",
                              inventory="https://cloud.comfy.org/api/object_info")
    assert backend.ready()
    assert calls[0][1] == {"Authorization": "Bearer test-key"}
    assert calls[1][1] == {"X-API-Key": "test-key"}


def test_managed_v2_endpoint_does_not_guess_native_inventory(backend, monkeypatch):
    monkeypatch.setenv("COMFY_BASE_URL", "https://example.run.comfy.app")
    calls = install_responses(monkeypatch, node_info(), base="https://example.run.comfy.app")
    status = backend.preflight()
    assert status.connected and not status.ready
    assert status.status == "unverified"
    assert "COMFYUI_BASE_URL" in status.message
    assert len(calls) == 1


def test_explicit_native_endpoint_uses_its_own_credentials(backend, monkeypatch):
    monkeypatch.setenv("COMFY_BASE_URL", "https://example.run.comfy.app")
    monkeypatch.setenv("COMFY_API_KEY", "proxy-key")
    monkeypatch.setenv("COMFYUI_BASE_URL", "https://inventory.example/comfy")
    monkeypatch.setenv("COMFYUI_API_KEY", "native-key")
    calls = install_responses(monkeypatch, node_info(), base="https://example.run.comfy.app",
                              inventory="https://inventory.example/comfy/object_info")
    assert backend.ready()
    assert calls[1][1] == {"Authorization": "Bearer native-key"}


def test_failed_check_recovers_after_cache_expiry(backend, monkeypatch):
    from cuesync import readiness

    clock = [100.0]
    monkeypatch.setattr(readiness.time, "monotonic", lambda: clock[0])
    info = node_info(models=())
    calls = install_responses(monkeypatch, info)
    assert not backend.ready()
    info["UNETLoader"]["input"]["required"]["unet_name"][0].append("fire.safetensors")
    assert not backend.ready()
    assert len(calls) == 2
    clock[0] += 16
    assert backend.ready()
    assert len(calls) == 4


def test_workflow_and_credentials_changes_invalidate_cache(backend, monkeypatch):
    calls = install_responses(monkeypatch, node_info())
    assert backend.ready()
    monkeypatch.setenv("COMFY_API_KEY", "new-key")
    assert backend.ready()
    assert len(calls) == 4
    graph = json.loads(backend.workflow.read_text())
    graph["1"]["inputs"]["unet_name"] = "missing.safetensors"
    backend.workflow.write_text(json.dumps(graph))
    assert not backend.ready()
    assert "missing.safetensors" in backend.preflight().message


def test_network_errors_are_actionable_and_do_not_echo_secrets(backend, monkeypatch):
    from cuesync import readiness

    def unavailable(url, headers):
        raise urllib.error.URLError("secret-token must not appear")

    monkeypatch.setattr(readiness, "_fetch_json", unavailable)
    status = backend.preflight()
    assert not status.connected and not status.ready
    assert status.status == "unreachable"
    assert "COMFY_BASE_URL" in status.message
    assert "secret-token" not in str(status)


def test_inventory_failure_distinguishes_connected_proxy(backend, monkeypatch):
    from cuesync import readiness

    def unavailable(url, headers):
        if url.endswith("/api/v2/jobs?limit=1"):
            return {"jobs": []}
        raise TimeoutError()

    monkeypatch.setattr(readiness, "_fetch_json", unavailable)
    status = backend.preflight()
    assert status.connected and not status.ready
    assert status.status == "inventory_unavailable"


def test_malformed_inventory_does_not_assert_all_models_exist(backend, monkeypatch):
    info = node_info()
    info["UNETLoader"]["input"]["required"]["unet_name"] = ["COMBO", {}]
    install_responses(monkeypatch, info)
    status = backend.preflight()
    assert not status.ready and status.status == "unverified"
    assert "UNETLoader.unet_name" in status.message


def test_model_paths_are_normalized_but_not_matched_by_basename(backend, monkeypatch):
    graph = json.loads(backend.workflow.read_text())
    graph["1"]["inputs"]["unet_name"] = "fire/model.safetensors"
    backend.workflow.write_text(json.dumps(graph))
    install_responses(monkeypatch, node_info(models=("fire\\model.safetensors",)))
    assert backend.ready()
    graph["1"]["inputs"]["unet_name"] = "other/model.safetensors"
    backend.workflow.write_text(json.dumps(graph))
    assert not backend.ready()


def test_actual_shipped_workflow_requires_all_five_weight_files(backend, monkeypatch):
    backend.workflow = Path("workflows/fluid_element.api.json")
    graph = json.loads(backend.workflow.read_text())
    info = {node["class_type"]: {"input": {"required": {}}} for node in graph.values()}
    install_responses(monkeypatch, info)
    status = backend.preflight()
    assert not status.ready
    for field in ("unet_name", "clip_name", "vae_name", "lora_name"):
        assert field in status.message


def test_actual_workflow_names_each_missing_weight(backend, monkeypatch):
    backend.workflow = Path("workflows/fluid_element.api.json")
    graph = json.loads(backend.workflow.read_text())
    info = {node["class_type"]: {"input": {"required": {}}} for node in graph.values()}
    for class_name, field in (("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"),
                              ("VAELoader", "vae_name"), ("LTXICLoRALoaderModelOnly", "lora_name")):
        info[class_name]["input"]["required"][field] = [[], {}]
    install_responses(monkeypatch, info)
    status = backend.preflight()
    assert status.missing_models == (
        "CLIPLoader.clip_name: gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "LTXICLoRALoaderModelOnly.lora_name: ainvfx-fluid.safetensors",
        "UNETLoader.unet_name: ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "VAELoader.vae_name: ltx-2.5-audio-vae-bf16.safetensors",
        "VAELoader.vae_name: ltx-2.5-video-vae-bf16.safetensors",
    )


def test_new_combo_schema_can_confirm_available_weights(backend, monkeypatch):
    info = node_info()
    info["UNETLoader"]["input"]["required"]["unet_name"] = ["COMBO", {"options": ["fire.safetensors"]}]
    install_responses(monkeypatch, info)
    assert backend.ready()


def test_endpoint_change_is_checked_before_cached_success_expires(backend, monkeypatch):
    install_responses(monkeypatch, node_info())
    assert backend.ready()
    monkeypatch.setenv("COMFY_BASE_URL", "https://changed.example")
    install_responses(monkeypatch, node_info(), base="https://changed.example")
    assert backend.preflight().status == "unverified"


@pytest.mark.parametrize("url", ["ftp://localhost", "http://localhost:broken", "http://user:secret@localhost",
                                "http://localhost?key=secret", "http://localhost/#secret"])
def test_invalid_connection_urls_are_rejected_without_echo(backend, monkeypatch, url):
    monkeypatch.setenv("COMFY_BASE_URL", url)
    status = backend.preflight()
    assert not status.ready and status.status == "invalid_config"
    assert "secret" not in str(status)


def test_missing_or_malformed_workflow_is_actionable(backend, monkeypatch):
    backend.workflow.write_text('{"nodes": []}')
    assert backend.preflight().status == "invalid_workflow"
    backend.workflow.unlink()
    assert backend.preflight().status == "invalid_workflow"


@pytest.fixture
def native_server():
    requests = []
    state = {"deny": False, "redirect": False}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            if state["redirect"]:
                self.send_response(302)
                self.send_header("Location", "/credential-target")
                self.end_headers()
                return
            if state["deny"]:
                self.send_response(401)
                self.end_headers()
                return
            expected_key = "Bearer proxy-key" if self.path.startswith("/api/v2/jobs") else "Bearer native-key"
            if self.headers.get("Authorization") != expected_key:
                self.send_response(403)
                self.end_headers()
                return
            payload = {"jobs": []} if self.path == "/api/v2/jobs?limit=1" else node_info()
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_preflight_over_http_uses_read_only_authenticated_endpoints(backend, monkeypatch, native_server):
    url, requests, _ = native_server
    monkeypatch.setenv("COMFY_BASE_URL", url)
    monkeypatch.setenv("COMFYUI_BASE_URL", url)
    monkeypatch.setenv("COMFY_API_KEY", "proxy-key")
    monkeypatch.setenv("COMFYUI_API_KEY", "native-key")
    assert backend.ready()
    assert requests == [("/api/v2/jobs?limit=1", "Bearer proxy-key"), ("/object_info", "Bearer native-key")]


def test_http_authentication_failure_is_not_gpu_ready(backend, monkeypatch, native_server):
    url, _, state = native_server
    state["deny"] = True
    monkeypatch.setenv("COMFY_BASE_URL", url)
    status = backend.preflight()
    assert status.status == "authentication_failed"
    assert not status.ready and "COMFY_API_KEY" in status.message


def test_http_redirect_cannot_forward_api_credentials(backend, monkeypatch, native_server):
    url, requests, state = native_server
    state["redirect"] = True
    monkeypatch.setenv("COMFY_BASE_URL", url)
    monkeypatch.setenv("COMFY_API_KEY", "proxy-key")
    assert not backend.ready()
    assert requests == [("/api/v2/jobs?limit=1", "Bearer proxy-key")]


def test_graph_without_required_renderer_inputs_is_not_ready(backend, monkeypatch):
    backend.workflow.write_text(json.dumps({"1": {"class_type": "SaveVideo", "inputs": {}}}))
    calls = install_responses(monkeypatch, node_info())
    status = backend.preflight()
    assert not status.ready and status.status == "invalid_workflow"
    assert "LoadVideo" in status.message
    assert calls == []


def test_duplicate_scene_adapter_nodes_only_disable_scene_render(backend, monkeypatch):
    graph = json.loads(backend.workflow.read_text())
    graph["7"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {}}
    backend.workflow.write_text(json.dumps(graph))
    install_responses(monkeypatch, node_info())
    status = backend.preflight()
    assert status.ready and not status.scene_ready


def test_missing_node_inputs_cannot_crash_graph_binding(backend, monkeypatch):
    graph = json.loads(backend.workflow.read_text())
    del graph["6"]["inputs"]
    backend.workflow.write_text(json.dumps(graph))
    calls = install_responses(monkeypatch, node_info())
    assert backend.preflight().status == "invalid_workflow"
    assert calls == []


@pytest.mark.parametrize("inventory_failure", [False, True])
def test_truncated_http_body_is_reported_as_unavailable(backend, monkeypatch, inventory_failure):
    from cuesync import readiness

    def incomplete(url, headers):
        if inventory_failure and url.endswith("/api/v2/jobs?limit=1"):
            return {"jobs": []}
        raise http.client.IncompleteRead(b"{", 10)

    monkeypatch.setattr(readiness, "_fetch_json", incomplete)
    status = backend.preflight()
    assert not status.ready
    assert status.connected == inventory_failure
    assert status.status == ("inventory_unavailable" if inventory_failure else "unreachable")
