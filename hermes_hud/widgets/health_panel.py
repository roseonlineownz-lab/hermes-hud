"""Health panel — API keys, services, system status."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static

from ..collectors.health import HealthState

# Provider → primary API key name
_PROVIDER_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "xai": "XAI_API_KEY",
    "grok": {"XAI_API_KEY", "GROK_API_KEY"},
    "gemini": "GEMINI_API_KEY",
    "qwen": {"QWEN_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_SECRET"},
    "deepseek": "DEEPSEEK_API_KEY",
    "volcengine": {"QWEN_API_KEY", "DASHSCOPE_API_KEY"},
}
_ALWAYS_CRITICAL = {"ANTHROPIC_API_KEY"}


def _critical_keys(provider: str) -> set[str]:
    """Return the set of key names considered critical for the given provider."""
    provider_normalized = provider.lower()
    keys = _PROVIDER_KEY_MAP.get(provider_normalized, {"ANTHROPIC_API_KEY"})

    if isinstance(keys, str):
        keys_set = {keys}
    else:
        keys_set = set(keys)

    if not keys_set:
        keys_set = {"ANTHROPIC_API_KEY"}

    return keys_set | _ALWAYS_CRITICAL


class HealthPanel(Static):
    """Panel showing system health status."""

    DEFAULT_CSS = """
    HealthPanel {
        height: auto;
        padding: 1 2;
    }
    """

    def __init__(self, health: HealthState, **kwargs):
        super().__init__(**kwargs)
        self.health = health

    def compose(self) -> ComposeResult:
        h = self.health

        # Overall status
        if h.all_healthy:
            yield Static("[bold green]⚿ SYSTEM HEALTH — ALL CRITICAL OK[/bold green]")
        else:
            problems = h.required_keys_missing + h.services_required_missing
            yield Static(f"[bold yellow]⚿ SYSTEM HEALTH — {problems} CRITICAL ISSUE{'S' if problems != 1 else ''}[/bold yellow]")
        yield Static("")

        # Model & Provider
        yield Static(
            f"  Model: [bold]{h.config_provider}/{h.config_model}[/bold]"
        )
        db_size = f"{h.state_db_size / 1024 / 1024:.1f} MB" if h.state_db_size else "?"
        yield Static(
            f"  State DB: {'[green]exists[/green]' if h.state_db_exists else '[red]missing[/red]'}"
            f" ({db_size})"
        )
        yield Static("")

        # API Keys
        yield Static("  [bold underline]API Keys[/bold underline]")
        for key in h.keys:
            if key.present:
                req = "[blue]" if not key.required else ""
                req_end = "[/blue]" if not key.required else ""
                yield Static(f"  {req}[green]✔ {key.name}[/green]{req_end}")
            else:
                color = "yellow" if not key.required else "red"
                note = f" — {key.note}" if key.note else ""
                yield Static(f"  [{color}]✗ {key.name}[/{color}][dim]{note}[/dim]")
        yield Static(
            f"  [dim]{h.keys_ok} configured, {h.keys_missing} missing | required: {h.required_keys_ok}/{h.required_keys_total}[/dim]"
        )
        yield Static("")

        # Services
        yield Static("  [bold underline]Services[/bold underline]")
        for svc in h.services:
            if svc.running:
                req = "[blue]" if not svc.required else ""
                req_end = "[/blue]" if not svc.required else ""
                pid_str = f" (pid {svc.pid})" if svc.pid else ""
                yield Static(f"  {req}[green]✔ {svc.name}{pid_str}[green]{req_end}")
            else:
                color = "yellow" if not svc.required else "red"
                note = f" — {svc.note}" if svc.note else ""
                yield Static(f"  [{color}]✗ {svc.name}[/{color}][dim]{note}[/dim]")
        yield Static(
            f"  [dim]services: {h.services_ok}/{len(h.services)} | critical OK: {h.services_required_ok}/{h.services_required_total}[/dim]"
        )
        yield Static("")

        # Quick diagnostics
        yield Static("  [bold underline]Diagnostics[/bold underline]")

        if not h.hermes_dir_exists:
            yield Static("  [red bold]✗ ~/.hermes directory not found![/red bold]")

        critical = _critical_keys(h.config_provider)
        missing_critical = []
        missing_optional = []
        for k in h.keys:
            if not k.present:
                if k.name in critical:
                    missing_critical.append(k)
                else:
                    missing_optional.append(k)

        if missing_critical:
            for k in missing_critical:
                yield Static(f"  [red bold]⚠ {k.name} missing — critical for current provider[/red bold]")
        if missing_optional:
            names = ", ".join(k.name for k in missing_optional)
            yield Static(f"  [blue]◐ Optional keys not set: {names}[/blue]")

        dead_services = [s for s in h.services if s.required and not s.running]
        if dead_services:
            for s in dead_services:
                yield Static(f"  [red]◐ {s.name} required service down[/red]")

        if h.all_healthy:
            yield Static("  [green]Core systems nominal. Optional items can be enabled when needed.[/green]")
        else:
            yield Static("  [yellow]Core stack has dependency issues. Check above items.[/yellow]")
