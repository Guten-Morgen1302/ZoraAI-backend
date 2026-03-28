"""Docker-backed disposable URL sandbox executor."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any

DEFAULT_DOCKER_IMAGE = "url-sandbox"
DEFAULT_DOCKER_MEMORY = "512m"
DEFAULT_DOCKER_CPUS = "1.0"
DEFAULT_DOCKER_PIDS = "256"
DEFAULT_TIMEOUT_SEC = 45
DEBUG_ENABLED = (os.getenv("URL_SANDBOX_DEBUG", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}

logger = logging.getLogger("zora.url_analysis.docker_sandbox")


def _debug(message: str) -> None:
    line = f"[docker-sandbox] {message}"
    print(line)
    if DEBUG_ENABLED:
        logger.debug(line)


def _safe_output(initial_url: str = "") -> dict[str, Any]:
    return {
        "status": "error",
        "error": "",
        "error_stage": "",
        "initial_url": initial_url,
        "final_url": "",
        "redirect_chain": [],
        "dom_length": 0,
        "dom_metrics": {"node_count": 0, "dom_length": 0},
        "raw_html": "",
        "num_scripts": 0,
        "external_js": [],
        "requests": [],
        "network_requests": [],
        "external_domains": [],
        "third_party_domains": [],
        "suspicious_endpoints": [],
        "set_cookie_headers": [],
        "set_cookie_details": [],
        "cookies": [],
        "console_messages": [],
        "blocked_internal_requests": [],
        "request_failures": [],
        "suspicious_activity_flags": [],
    }


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(str(value or "").strip())
    except (TypeError, ValueError):
        return default


def _runtime_config() -> dict[str, Any]:
    return {
        "image": os.getenv("URL_SANDBOX_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE).strip() or DEFAULT_DOCKER_IMAGE,
        "memory": os.getenv("URL_SANDBOX_DOCKER_MEMORY", DEFAULT_DOCKER_MEMORY).strip() or DEFAULT_DOCKER_MEMORY,
        "cpus": os.getenv("URL_SANDBOX_DOCKER_CPUS", DEFAULT_DOCKER_CPUS).strip() or DEFAULT_DOCKER_CPUS,
        "pids": os.getenv("URL_SANDBOX_DOCKER_PIDS", DEFAULT_DOCKER_PIDS).strip() or DEFAULT_DOCKER_PIDS,
        "timeout_sec": max(15, min(_to_int(os.getenv("URL_SANDBOX_DOCKER_TIMEOUT_SEC"), DEFAULT_TIMEOUT_SEC), 120)),
    }


def _find_first_json_line(stdout: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _normalize_payload(payload: dict[str, Any], initial_url: str) -> dict[str, Any]:
    out = _safe_output(initial_url)
    out.update({k: v for k, v in payload.items() if k in out or k in {"status", "error", "error_stage"}})

    requests = out.get("requests")
    network_requests = out.get("network_requests")

    if isinstance(requests, list) and not isinstance(network_requests, list):
        out["network_requests"] = requests
    if isinstance(network_requests, list) and not isinstance(requests, list):
        out["requests"] = network_requests

    if not isinstance(out.get("requests"), list):
        out["requests"] = []
    if not isinstance(out.get("network_requests"), list):
        out["network_requests"] = []

    status = str(out.get("status", "") or "").strip().lower()
    if status not in {"success", "timeout", "error"}:
        status = "error"
    out["status"] = status

    if status == "success":
        out["error"] = ""
        out["error_stage"] = ""
    else:
        out["error"] = str(out.get("error") or "docker_sandbox_failure")
        out["error_stage"] = str(out.get("error_stage") or "docker_execution")

    return out


def analyze_url_via_docker(url: str, timeout_ms: int) -> dict[str, Any]:
    output = _safe_output(url)
    _debug(f"start url={url} timeout_ms={timeout_ms}")

    docker_bin = shutil.which("docker")
    if not docker_bin:
        _debug("docker binary not found in PATH")
        output["status"] = "error"
        output["error"] = "docker_binary_not_found"
        output["error_stage"] = "docker_binary_check"
        return output

    cfg = _runtime_config()
    cmd = [
        docker_bin,
        "run",
        "--rm",
        f"--memory={cfg['memory']}",
        f"--cpus={cfg['cpus']}",
        f"--pids-limit={cfg['pids']}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--tmpfs",
        "/home/sandbox/.cache:rw,noexec,nosuid,size=64m",
        str(cfg["image"]),
        str(url),
        "--timeout-ms",
        str(timeout_ms),
    ]
    _debug(f"running image={cfg['image']} memory={cfg['memory']} cpus={cfg['cpus']} pids={cfg['pids']}")

    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=int(cfg["timeout_sec"]),
        )
    except subprocess.TimeoutExpired:
        _debug("docker run timed out")
        output["status"] = "timeout"
        output["error"] = "docker_run_timeout"
        output["error_stage"] = "docker_run"
        return output
    except Exception as exc:  # noqa: BLE001
        _debug(f"docker run failed: {exc.__class__.__name__}:{exc}")
        output["status"] = "error"
        output["error"] = f"docker_run_failed:{exc.__class__.__name__}:{exc}"
        output["error_stage"] = "docker_run"
        return output

    parsed = _find_first_json_line(proc.stdout)
    if parsed is not None:
        _debug(f"received JSON payload from container exit={proc.returncode}")
        normalized = _normalize_payload(parsed, initial_url=url)
        if proc.returncode != 0 and normalized.get("status") == "success":
            normalized["status"] = "error"
            normalized["error"] = f"docker_exit_nonzero:{proc.returncode}"
            normalized["error_stage"] = "docker_run"
        return normalized

    _debug(f"invalid JSON from container exit={proc.returncode}")
    output["status"] = "error"
    output["error"] = f"docker_invalid_json_output:exit={proc.returncode}"
    output["error_stage"] = "docker_parse_output"
    stderr = (proc.stderr or "").strip()
    if stderr:
        output["console_messages"] = [stderr[:500]]
    return output
