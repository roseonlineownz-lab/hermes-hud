"""Health check collector — API keys, services, connectivity."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .utils import default_hermes_dir


@dataclass
class KeyStatus:
    name: str
    source: str  # env, auth.json, config
    present: bool = False
    required: bool = False
    note: str = ""


@dataclass
class ServiceStatus:
    name: str
    running: bool = False
    required: bool = True
    pid: Optional[int] = None
    note: str = ""


@dataclass
class HealthState:
    keys: list[KeyStatus] = field(default_factory=list)
    services: list[ServiceStatus] = field(default_factory=list)
    config_model: str = ""
    config_provider: str = ""
    hermes_dir_exists: bool = False
    state_db_exists: bool = False
    state_db_size: int = 0

    @property
    def keys_ok(self) -> int:
        return sum(1 for k in self.keys if k.present)

    @property
    def keys_missing(self) -> int:
        return sum(1 for k in self.keys if not k.present)

    @property
    def required_keys_ok(self) -> int:
        return sum(1 for k in self.keys if k.required and k.present)

    @property
    def required_keys_total(self) -> int:
        return sum(1 for k in self.keys if k.required)

    @property
    def required_keys_missing(self) -> int:
        return self.required_keys_total - self.required_keys_ok

    @property
    def services_ok(self) -> int:
        return sum(1 for s in self.services if s.running)

    @property
    def services_required_ok(self) -> int:
        return sum(1 for s in self.services if (not s.required) or s.running)

    @property
    def services_required_total(self) -> int:
        return sum(1 for s in self.services if s.required)

    @property
    def services_required_missing(self) -> int:
        return sum(1 for s in self.services if s.required and not s.running)

    @property
    def services_missing(self) -> int:
        return sum(1 for s in self.services if not s.running)

    @property
    def services_total(self) -> int:
        return len(self.services)

    @property
    def all_healthy(self) -> bool:
        return self.required_keys_missing == 0 and self.services_required_missing == 0


# Known API keys to check
# tuple: (name, source, note, required)
EXPECTED_KEYS = [
    ("ANTHROPIC_API_KEY", "env", "Primary LLM provider", True),
    ("OPENAI_API_KEY", "env", "OpenAI models", False),
    ("OPENROUTER_API_KEY", "env", "OpenRouter fallback provider", False),
    ("GEMINI_API_KEY", "env", "Gemini models", False),
    ("XAI_API_KEY", "env", "xAI / Grok models", False),
    ("QWEN_API_KEY", "env", "Qwen models", False),
    ("DASHSCOPE_API_KEY", "env", "Alibaba Cloud Qwen models", False),
    ("DEEPSEEK_API_KEY", "env", "DeepSeek models", False),
    ("TELEGRAM_BOT_TOKEN", "env", "Telegram bot notifications", False),
    ("DISCORD_BOT_TOKEN", "env", "Discord bot notifications", False),
    ("ELEVENLABS_API_KEY", "env", "TTS/voice provider", False),
    ("FIREWORKS_API_KEY", "env", "Fireworks provider", False),
    ("BROWSERBASE_API_KEY", "env", "Browser automation", False),
    ("GITHUB_TOKEN", "env", "GitHub API", False),
    ("KAGGLE_USERNAME", "env", "Kaggle username", False),
    ("KAGGLE_KEY", "env", "Kaggle API key", False),
    ("HOSTINGER_API_KEY", "env", "Hostinger API token", False),
    ("HOSTINGER_TOKEN", "env", "Hostinger API token", False),
    ("QWEN_API_SECRET", "env", "Qwen/DashScope secret", False),
    ("GROK_API_KEY", "env", "Grok alias", False),
]


def _load_dotenv_keys(dotenv_path: str) -> set[str]:
    """Load key names from a .env file (not values)."""
    keys = set()
    try:
        with open(dotenv_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key = line.split("=", 1)[0].strip()
                    if key:
                        keys.add(key)
    except (OSError, PermissionError):
        pass
    return keys


def _get_dotenv_keys(hermes_dir: str) -> set[str]:
    """Get all key names from hermes .env files."""
    keys: set[str] = set()
    for env_path in [
        os.path.join(hermes_dir, ".env"),
        os.path.expanduser("~/.env"),
    ]:
        keys.update(_load_dotenv_keys(env_path))
    return keys


def _check_env_key(name: str, dotenv_keys: set[str]) -> bool:
    """Check if a key is set in environment or .env files."""
    if os.environ.get(name, ""):
        return True
    return name in dotenv_keys


def _check_process(name: str, pattern: str, required: bool = True) -> ServiceStatus:
    """Check if a process matching pattern is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True, text=True, timeout=5,
        )
        pids = [int(p) for p in result.stdout.strip().split("\n") if p.strip()]
        if pids:
            return ServiceStatus(name=name, running=True, required=required, pid=pids[0])
        return ServiceStatus(name=name, running=False, required=required)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return ServiceStatus(name=name, running=False, required=required, note="check failed")


def _check_pid_file(name: str, pid_file: Path, required: bool = True) -> ServiceStatus:
    """Check if a PID file exists and the process is alive."""
    if not pid_file.exists():
        return ServiceStatus(name=name, running=False, required=required, note="no pid file")

    try:
        data = json.loads(pid_file.read_text())
        pid = data.get("pid")
        if pid:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "pid="],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return ServiceStatus(name=name, running=True, required=required, pid=pid)
            return ServiceStatus(name=name, running=False, required=required, pid=pid, note="pid file exists but process dead")
    except (json.JSONDecodeError, OSError, subprocess.TimeoutExpired):
        pass

    return ServiceStatus(name=name, running=False, required=required, note="pid file unreadable")


def _check_systemd_service(name: str, service: str, required: bool = True) -> ServiceStatus:
    """Check systemd user service status."""
    env = os.environ.copy()
    runtime_dir = env.get("XDG_RUNTIME_DIR")
    if not runtime_dir:
        candidate = Path(f"/run/user/{os.getuid()}")
        if candidate.is_dir():
            runtime_dir = str(candidate)
            env["XDG_RUNTIME_DIR"] = runtime_dir

    if runtime_dir and not env.get("DBUS_SESSION_BUS_ADDRESS"):
        bus_path = Path(runtime_dir) / "bus"
        if bus_path.exists():
            env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus_path}"

    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", service],
            capture_output=True, text=True, timeout=5, env=env,
        )
        is_active = result.stdout.strip() == "active"
        note = result.stdout.strip() or result.stderr.strip()
        if not note:
            note = "inactive"
        return ServiceStatus(name=name, running=is_active, required=required, note=note)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ServiceStatus(name=name, running=False, required=required, note="systemctl unavailable")


def _check_http(name: str, url: str, required: bool = True, expected: tuple[int, ...] = (200,), timeout: int = 2) -> ServiceStatus:
    """Check a local HTTP endpoint."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hermes-hud-health"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            code = response.getcode()
            if code in expected:
                return ServiceStatus(name=name, running=True, required=required, note=f"HTTP {code}")
            return ServiceStatus(name=name, running=False, required=required, note=f"HTTP {code}")
    except urllib.error.HTTPError as exc:
        if exc.code in expected:
            return ServiceStatus(name=name, running=True, required=required, note=f"HTTP {exc.code}")
        return ServiceStatus(name=name, running=False, required=required, note=f"HTTP {exc.code}")
    except Exception as exc:
        return ServiceStatus(name=name, running=False, required=required, note=f"{exc.__class__.__name__}: {exc}")


def _check_tcp(name: str, host: str, port: int, required: bool = True, timeout: int = 2) -> ServiceStatus:
    """Check basic TCP reachability."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return ServiceStatus(name=name, running=True, required=required, note="tcp ok")
    except Exception as exc:
        return ServiceStatus(name=name, running=False, required=required, note=f"tcp fail: {exc}")


def collect_health(hermes_dir: str | None = None) -> HealthState:
    """Collect health status."""
    if hermes_dir is None:
        hermes_dir = default_hermes_dir(hermes_dir)

    hermes_path = Path(hermes_dir)
    state = HealthState()

    # Directory checks
    state.hermes_dir_exists = hermes_path.exists()
    state_db = hermes_path / "state.db"
    state.state_db_exists = state_db.exists()
    if state.state_db_exists:
        try:
            state.state_db_size = state_db.stat().st_size
        except OSError:
            pass

    # Config — reuse the config collector
    from .config import collect_config
    try:
        config = collect_config(hermes_dir)
        state.config_model = config.model
        state.config_provider = config.provider
    except Exception:
        pass

    # API keys
    dotenv_keys = _get_dotenv_keys(hermes_dir)

    known_names = {key_name for key_name, _, _, _ in EXPECTED_KEYS}
    for key_name, source, note, required in EXPECTED_KEYS:
        present = _check_env_key(key_name, dotenv_keys)
        state.keys.append(KeyStatus(
            name=key_name,
            source=source,
            present=present,
            required=required,
            note=note if not present else "",
        ))

    # Auto-discover additional keys from .env for visibility.
    for extra_key in sorted(dotenv_keys):
        if extra_key not in known_names:
            if any(extra_key.endswith(suffix) for suffix in ("_API_KEY", "_TOKEN", "_SECRET", "_KEY")):
                state.keys.append(KeyStatus(
                    name=extra_key,
                    source="env",
                    present=True,
                    required=False,
                    note="discovered",
                ))

    # Core system services
    state.services.append(_check_systemd_service("Gateway (systemd)", "hermes-gateway", required=True))
    state.services.append(_check_systemd_service("OpenClaw Gateway", "openclaw-gateway", required=True))
    state.services.append(_check_systemd_service("OpenClaw Node", "openclaw-node", required=False))
    state.services.append(_check_systemd_service("ClawMem Serve", "clawmem-serve", required=True))
    state.services.append(_check_systemd_service("Lead API", "nova-lead-api", required=True))
    state.services.append(_check_systemd_service("Aion UI", "aionui-autologin", required=False))
    state.services.append(_check_systemd_service("Aion WebUI", "aionui-webui", required=False))
    state.services.append(_check_systemd_service("Hermes Office Adapter", "hermes-office-adapter", required=False))
    state.services.append(_check_systemd_service("Hermes Office Dev", "hermes-office-dev", required=False))
    state.services.append(_check_systemd_service("Jarvis Bridge", "jarvis-bridge-7777", required=False))
    state.services.append(_check_systemd_service("Jarvis API", "jarvis-api", required=False))
    state.services.append(_check_systemd_service("qdrant", "qdrant", required=False))
    state.services.append(_check_systemd_service("metaclaw", "metaclaw", required=False))
    state.services.append(_check_systemd_service("Tabby", "tabby", required=False))
    state.services.append(_check_systemd_service("n8n", "n8n", required=False))
    state.services.append(_check_tcp("Langfuse", "127.0.0.1", 3099, required=False))
    state.services.append(_check_systemd_service("comfyui", "comfyui", required=False))

    # Process checks for key runtime daemons
    state.services.append(_check_process("vibevoice", "vibevoice"))
    state.services.append(_check_process("vibevoice-bridge", "vibevoice-bridge"))
    state.services.append(_check_process("kokoro-tts", "kokoro-server/server.py", required=False))
    state.services.append(_check_pid_file("Telegram Gateway", hermes_path / "gateway.pid", required=False))

    # Endpoint checks (best signal for false-positive service checks)
    state.services.append(_check_http("Hermes API", "http://127.0.0.1:8643/health", required=True, expected=(200,)))
    state.services.append(_check_http("Hermes API (alt)", "http://127.0.0.1:8644/", required=False, expected=(200, 302, 307)))
    state.services.append(_check_http("OpenClaw Gateway", "http://127.0.0.1:18793/health", required=True, expected=(200,)))
    state.services.append(_check_http("Claw3D Backend", "http://127.0.0.1:8095/health", required=False, expected=(200,)))
    state.services.append(_check_http("VibeVoice SVC", "http://127.0.0.1:8093/health", required=False, expected=(200,)))
    state.services.append(_check_http("VibeVoice Bridge", "http://127.0.0.1:8094/health", required=False, expected=(200,)))
    state.services.append(_check_http("Kokoro TTS", "http://127.0.0.1:8098/health", required=False, expected=(200, 405, 404)))
    state.services.append(_check_http("N8N", "http://127.0.0.1:5678/health", required=False, expected=(200, 404, 307)))
    state.services.append(_check_http("Noiz OCR", "http://127.0.0.1:8096/health", required=False, expected=(200,)))
    state.services.append(_check_http("Langfuse", "http://127.0.0.1:3099/api/public/health", required=False, expected=(200,)))
    state.services.append(_check_http("ComfyUI", "http://127.0.0.1:8188", required=False, expected=(200, 307, 302, 404)))
    state.services.append(_check_http("Lead API", "http://127.0.0.1:8099/health", required=True, expected=(200,)))
    state.services.append(_check_http("Office", "http://127.0.0.1:9120", required=False, expected=(200, 301, 302, 307)))
    state.services.append(_check_http("Office WS", "http://127.0.0.1:18800", required=False, expected=(200,)))
    state.services.append(_check_http("Aion UI", "http://127.0.0.1:3000", required=False, expected=(200, 302, 307)))
    state.services.append(_check_http("Aion UI Alt", "http://127.0.0.1:3001", required=False, expected=(200, 302, 307)))

    # Local infra ports
    state.services.append(_check_tcp("Redis", "127.0.0.1", 6379, required=False))
    state.services.append(_check_tcp("PostgreSQL", "127.0.0.1", 5432, required=False))

    # Optional remote VPS status (if configured)
    vps_host = (
        os.environ.get("NOVA_VPS_HOST")
        or os.environ.get("HERMES_HUD_VPS")
        or os.environ.get("HERMES_HUD_REMOTE")
    )
    if vps_host:
        state.services.append(_check_tcp(f"VPS Lead API ({vps_host}:8099)", vps_host, 8099, required=False))
        state.services.append(_check_tcp(f"VPS OpenClaw ({vps_host}:18793)", vps_host, 18793, required=False))
        state.services.append(_check_tcp(f"VPS Gateway ({vps_host}:8643)", vps_host, 8643, required=False))
        state.services.append(_check_tcp(f"VPS Office WS ({vps_host}:18800)", vps_host, 18800, required=False))
        state.services.append(_check_tcp(f"VPS n8n ({vps_host}:5678)", vps_host, 5678, required=False))
        state.services.append(_check_tcp(f"VPS Aion UI ({vps_host}:3000)", vps_host, 3000, required=False))
        state.services.append(_check_tcp(f"VPS Qdrant ({vps_host}:6333)", vps_host, 6333, required=False))
        state.services.append(_check_tcp(f"VPS Redis ({vps_host}:6379)", vps_host, 6379, required=False))
        state.services.append(_check_tcp(f"VPS PostgreSQL ({vps_host}:5432)", vps_host, 5432, required=False))

    return state
