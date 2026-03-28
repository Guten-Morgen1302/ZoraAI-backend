# URL Analysis Playwright Sandbox (Docker)

This folder contains a production-grade disposable URL sandbox container used by the URL analysis backend.

## What It Does

The container runs once per URL, captures telemetry, prints JSON to stdout, and exits.

Telemetry includes:
- final URL and redirect chain
- request stream (URL, method, resource type)
- external and third-party domains
- script count and external JS URLs
- cookie telemetry from browser context
- Set-Cookie headers and parsed attributes
- DOM metrics (node count, HTML size)
- optional console and request-failure logs
- suspicious activity flags

## Files

- `Dockerfile`: Python 3.11-slim + Playwright Chromium runtime
- `requirements.txt`: minimal Python dependencies for the sandbox
- `sandbox_runner.py`: CLI telemetry runner
- `.dockerignore`: build context hygiene

## Build

Run from this folder:

```bash
docker build -t url-sandbox .
```

## Run (single URL)

```bash
docker run --rm url-sandbox https://example.com
```

Equivalent explicit form:

```bash
docker run --rm url-sandbox python sandbox_runner.py https://example.com
```

With strict hardening:

```bash
docker run --rm \
  --memory=512m \
  --cpus=1.0 \
  --pids-limit=256 \
  --cap-drop=ALL \
  --security-opt=no-new-privileges:true \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --tmpfs /home/sandbox/.cache:rw,noexec,nosuid,size=64m \
  url-sandbox https://example.com
```

## Output Shape

The runner prints one JSON object to stdout:

- `status`: success, timeout, or error
- `error`, `error_stage`
- `initial_url`, `final_url`, `redirect_chain`
- `requests` and `network_requests`
- `external_domains`, `third_party_domains`
- `num_scripts`, `external_js`
- `cookies`, `set_cookie_headers`, `set_cookie_details`
- `dom_metrics`, `dom_length`
- `suspicious_activity_flags`

## Security Notes

- Chromium runs headless with GPU and extensions disabled.
- Internal IP / localhost literal requests are blocked in the browser route guard.
- Use Docker runtime controls from the command above for stronger process isolation.
- Keep this image dedicated to sandboxing only.

## Backend Integration

The backend can invoke this image per URL via Docker CLI and parse stdout JSON. Configure runtime using environment variables:

- `URL_SANDBOX_MODE=local|docker|auto`
- `URL_SANDBOX_DOCKER_IMAGE=url-sandbox`
- `URL_SANDBOX_DOCKER_MEMORY=512m`
- `URL_SANDBOX_DOCKER_CPUS=1.0`
- `URL_SANDBOX_DOCKER_PIDS=256`
- `URL_SANDBOX_DOCKER_TIMEOUT_SEC=45`

`auto` mode attempts Docker first and falls back to local Playwright execution if container execution fails.
