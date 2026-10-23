"""Read-only workflow preflight. Never submit a job to discover GPU readiness.

API v2's proxy health route checks only the proxy process. Native ComfyUI's
object_info supplies loaded node classes and each model loader's filename
choices. Cloud documents the same inventory at /api/object_info; v2-only
deployments do not promise an inventory route.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

CLOUD = "https://cloud.comfy.org"
CACHE_SECONDS = 15.0
MODEL_SUFFIXES = (".safetensors", ".ckpt", ".pth", ".pt", ".bin", ".gguf")


@dataclass(frozen=True)
class Readiness:
    ready: bool
    scene_ready: bool
    connected: bool
    status: str
    message: str
    missing_nodes: tuple[str, ...] = ()
    missing_models: tuple[str, ...] = ()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward either credential to a redirected origin.
        return None


def _fetch_json(url: str, headers: dict[str, str]) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json", **headers})
    with urllib.request.build_opener(_NoRedirect).open(request, timeout=2.5) as response:
        if response.status != 200:
            raise ValueError("unexpected response")
        body = response.read(20 * 1024 * 1024 + 1)
    if len(body) > 20 * 1024 * 1024:
        raise ValueError("inventory response too large")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("expected an object")
    return data


def _base_url(raw: str) -> str:
    """Validate without echoing URLs, which may contain credentials."""
    parsed = urlsplit(raw)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("invalid URL")
    _ = parsed.port
    return raw.rstrip("/")


def _cloud(base: str) -> bool:
    parsed = urlsplit(base)
    return (parsed.scheme == "https" and parsed.hostname == "cloud.comfy.org"
            and parsed.port in (None, 443) and not parsed.path.strip("/"))


def _inventory(base: str, native: str, key: str, native_key: str) -> tuple[str | None, dict[str, str]]:
    if _cloud(base):
        return base + "/api/object_info", {"X-API-Key": key}
    if native:
        return _base_url(native) + "/object_info", ({"Authorization": f"Bearer {native_key}"} if native_key else {})
    parsed = urlsplit(base)
    # This is the documented co-located/tunnelled setup. Do not infer upstream
    # ports for arbitrary hosts, custom proxies, or hosted v2 deployments.
    if (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")
            and parsed.port == 8189 and not parsed.path.strip("/")):
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        return urlunsplit(("http", f"{host}:8188", "/object_info", "", "")), {}
    return None, {}


def _model_choices(definition: dict, name: str) -> list | None:
    inputs = definition.get("input", {})
    if not isinstance(inputs, dict):
        return None
    for section in ("required", "optional"):
        fields = inputs.get(section, {})
        widget = fields.get(name) if isinstance(fields, dict) else None
        if isinstance(widget, (list, tuple)) and widget:
            if isinstance(widget[0], (list, tuple)):
                return list(widget[0])
            if len(widget) > 1 and isinstance(widget[1], dict) and isinstance(widget[1].get("options"), list):
                return widget[1]["options"]
    return None


def _resources(graph: dict, inventory: dict, scenes: bool) -> Readiness:
    classes = {node["class_type"] for node in graph.values()}
    missing_nodes = tuple(sorted(name for name in classes if not isinstance(inventory.get(name), dict)))
    missing_models, unknown = set(), set()
    for node in graph.values():
        name = node["class_type"]
        if name in missing_nodes:
            continue
        for field_name, value in node.get("inputs", {}).items():
            if not isinstance(value, str) or not value.lower().endswith(MODEL_SUFFIXES):
                continue
            label = f"{name}.{field_name}: {value}"
            choices = _model_choices(inventory[name], field_name)
            if choices is None:
                unknown.add(label)
            elif value.replace("\\", "/") not in {str(item).replace("\\", "/") for item in choices}:
                missing_models.add(label)
    missing_models = tuple(sorted(missing_models))
    if missing_nodes or missing_models:
        parts = (["Missing nodes: " + ", ".join(missing_nodes)] if missing_nodes else [])
        if missing_models:
            parts.append("Missing model files: " + "; ".join(missing_models))
        return Readiness(False, False, True, "missing_resources",
                         ". ".join(parts) + ". Install these resources in the target ComfyUI and refresh after 15 seconds.",
                         missing_nodes, missing_models)
    if unknown:
        return Readiness(False, False, True, "unverified",
                         "Model filenames could not be verified: " + "; ".join(sorted(unknown))
                         + ". The target must expose filename choices in object_info.")
    scene_nodes = sum(node["class_type"] == "LTXVImgToVideoInplace" for node in graph.values())
    scene_ready = scenes and scene_nodes == 1 and isinstance(inventory.get("LoadImage"), dict)
    message = "Required workflow nodes and model filenames are available; model loading and VRAM are not tested."
    if scenes and not scene_ready:
        message += " Scene rendering also needs exactly one LTXVImgToVideoInplace in the graph and the LoadImage node."
    return Readiness(True, scene_ready, True, "ready", message)


def _probe(workflow: bytes, base: str, key: str, native: str, native_key: str, scenes: bool) -> Readiness:
    from .runner import GraphBindError, bind_fluid_graph

    if not base and not key:
        return Readiness(False, False, False, "not_configured",
                         "Set COMFY_BASE_URL for a local Comfy API proxy, or COMFY_API_KEY for Comfy Cloud.")
    try:
        base = _base_url(base or CLOUD)
        inventory_url, inventory_headers = _inventory(base, native, key, native_key)
    except ValueError:
        return Readiness(False, False, False, "invalid_config",
                         "COMFY_BASE_URL and COMFYUI_BASE_URL must be HTTP(S) URLs without credentials, queries or fragments.")
    if _cloud(base) and not key:
        return Readiness(False, False, False, "not_configured", "Comfy Cloud requires COMFY_API_KEY.")
    try:
        graph = json.loads(workflow)
        if (not isinstance(graph, dict) or not graph or any(
                not isinstance(node, dict) or not isinstance(node.get("class_type"), str)
                or not isinstance(node.get("inputs"), dict) for node in graph.values())):
            raise ValueError("invalid graph")
    except (ValueError, UnicodeError):
        return Readiness(False, False, False, "invalid_workflow", "The configured workflow must be a nonempty API-format JSON graph.")
    try:
        bind_fluid_graph(graph, prompt="ainvfxfluid, preflight", seed=0)
    except GraphBindError as exc:
        return Readiness(False, False, False, "invalid_workflow", "The configured render graph is invalid: " + str(exc))
    try:
        # Unlike the proxy-only health route, this also verifies API credentials.
        response = _fetch_json(base + "/api/v2/jobs?limit=1", {"Authorization": f"Bearer {key}"} if key else {})
        if not isinstance(response.get("jobs"), list):
            raise ValueError("invalid jobs response")
    except urllib.error.HTTPError as exc:
        return Readiness(False, False, False, "authentication_failed" if exc.code in (401, 403) else "unreachable",
                         "Comfy API access failed (HTTP " + str(exc.code) + "). Check COMFY_BASE_URL and COMFY_API_KEY.")
    except (OSError, ValueError, http.client.HTTPException):
        return Readiness(False, False, False, "unreachable", "Comfy API could not be reached. Check COMFY_BASE_URL and the running proxy or deployment.")
    if inventory_url is None:
        return Readiness(False, False, True, "unverified",
                         "Comfy API is reachable, but this v2 endpoint has no documented node/model inventory. "
                         "Set COMFYUI_BASE_URL to this deployment's native ComfyUI for preflight; no GPU job was submitted.")
    try:
        inventory = _fetch_json(inventory_url, inventory_headers)
    except (OSError, ValueError, http.client.HTTPException):
        return Readiness(False, False, True, "inventory_unavailable",
                         "Comfy API is reachable, but node/model inventory could not be read. Check the native ComfyUI service "
                         "and COMFYUI_BASE_URL / COMFYUI_API_KEY (or COMFY_API_KEY for Cloud).")
    return _resources(graph, inventory, scenes)


@dataclass
class WorkflowPreflight:
    """Cache expensive inventory checks, invalidating config/file changes immediately."""

    _key: str = field(default="", init=False, repr=False)
    _checked_at: float = field(default=0.0, init=False, repr=False)
    _result: Readiness | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def check(self, workflow: Path, *, scenes: bool = True) -> Readiness:
        settings = tuple(os.environ.get(name, "").strip() for name in (
            "COMFY_BASE_URL", "COMFY_API_KEY", "COMFYUI_BASE_URL", "COMFYUI_API_KEY"))
        try:
            content = Path(workflow).read_bytes()
        except OSError:
            return Readiness(False, False, False, "invalid_workflow", "The configured workflow file could not be read.")
        # Keep only a digest in the cache: credentials never appear in repr/status.
        key = hashlib.sha256(json.dumps((settings, scenes)).encode() + b"\0" + content).hexdigest()
        with self._lock:
            if key != self._key or self._result is None or time.monotonic() - self._checked_at >= CACHE_SECONDS:
                self._result = _probe(content, *settings, scenes)
                self._key, self._checked_at = key, time.monotonic()
            return self._result
