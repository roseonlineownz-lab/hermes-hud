"""Regression tests for the medium-severity audit fixes.

4. _load_data must survive a collector failure (empty panel, not app crash).
5. collect_cron must tolerate alternate jobs.json shapes (bare list,
   null repeat/schedule/skills/string fields).
6. collect_config must normalize null/string toolsets and null sections
   so OverviewPanel's ', '.join never sees None.
7. build_timeline must not crash on an empty daily-stats date.
8. _check_pid_file must tolerate classic bare-integer pid files.
9. collect_sessions must skip rows with NULL started_at (no 1970 epoch
   sessions) and accept ISO-string timestamps.
"""

import asyncio
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path


# ── Fix 4: collector failure degrades gracefully ───────────────────────

def test_load_data_survives_collector_failure(env_override, monkeypatch):
    import hermes_hud.hud as hud_mod

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(hud_mod, "collect_cron", boom)
    monkeypatch.setattr(hud_mod, "collect_health", boom)
    monkeypatch.setenv("HERMES_HUD_NOBOOT", "1")

    async def run():
        app = hud_mod.HermesHUD()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("2")  # triggers _load_data
            await pilot.pause()
            assert app.is_running, "a failing collector must not crash the app"

    asyncio.run(run())


# ── Fix 5: cron jobs.json shape tolerance ───────────────────────────────

def test_collect_cron_tolerates_bare_list_and_nulls(tmp_path):
    from hermes_hud.collectors.cron import collect_cron

    cron_dir = tmp_path / "cron"
    cron_dir.mkdir()

    # bare list at top level, job with explicit nulls everywhere
    cron_dir.joinpath("jobs.json").write_text(json.dumps([
        {
            "id": None, "name": None, "prompt": None,
            "schedule_display": None, "schedule": None,
            "repeat": None, "skills": None, "deliver": None,
            "state": None, "enabled": True,
        },
        "not-a-dict",
    ]))

    state = collect_cron(str(tmp_path))
    assert state.total == 1
    job = state.jobs[0]
    assert job.name == "unnamed"
    assert job.schedule_display == "unknown"
    assert job.skills == []
    assert job.deliver == "local"


def test_collect_cron_tolerates_non_dict_top_level(tmp_path):
    from hermes_hud.collectors.cron import collect_cron

    cron_dir = tmp_path / "cron"
    cron_dir.mkdir()
    cron_dir.joinpath("jobs.json").write_text('"just a string"')

    assert collect_cron(str(tmp_path)).total == 0


# ── Fix 6: config toolsets/model normalization ──────────────────────────

def test_collect_config_normalizes_null_and_string_values(tmp_path):
    from hermes_hud.collectors.config import collect_config
    from hermes_hud.models import HUDState
    from hermes_hud.widgets import overview

    (tmp_path / "config.yaml").write_text(
        "model:\n"        # null section
        "toolsets:\n"     # null list
        "terminal:\n"
    )
    config = collect_config(str(tmp_path))
    assert config.toolsets == []
    assert config.model == ""
    assert config.provider == ""

    # OverviewPanel composes without TypeError on the joined toolsets
    recorded = []

    class Recorder:
        def __init__(self, content="", **kw):
            recorded.append(str(content))

    real = overview.Static
    overview.Static = Recorder
    try:
        list(overview.OverviewPanel(HUDState(config=config)).compose())
    finally:
        overview.Static = real
    assert recorded, "panel must render"

    # a scalar toolsets value becomes a one-element list
    (tmp_path / "config.yaml").write_text("toolsets: terminal\n")
    assert collect_config(str(tmp_path)).toolsets == ["terminal"]


# ── Fix 7: timeline survives an empty daily-stats date ──────────────────

def test_build_timeline_handles_empty_date():
    from hermes_hud.collectors.timeline import build_timeline
    from hermes_hud.models import DailyStats, HUDState, SessionInfo, SessionsState

    sessions = SessionsState(
        sessions=[SessionInfo(id="s1", source="cli", title="t",
                              started_at=datetime(2026, 6, 1), ended_at=None,
                              message_count=1, tool_call_count=0,
                              input_tokens=0, output_tokens=0)],
        daily_stats=[DailyStats(date="", sessions=1, messages=99, tool_calls=0)],
    )
    events = build_timeline(HUDState(sessions=sessions))  # must not raise
    assert not any(e.title.startswith("Most active day") for e in events)


# ── Fix 8: bare-integer pid files ────────────────────────────────────────

def test_check_pid_file_handles_bare_integer(tmp_path):
    from hermes_hud.collectors.health import _check_pid_file

    pid_file = tmp_path / "gateway.pid"
    pid_file.write_text("99999999\n")  # classic pid file: just the number

    status = _check_pid_file("Gateway", pid_file)  # must not raise
    assert status.running is False
    assert status.pid == 99999999

    pid_file.write_text('"nonsense"')  # valid JSON, wrong shape
    status = _check_pid_file("Gateway", pid_file)
    assert status.running is False


# ── Fix 9: session timestamp robustness ─────────────────────────────────

def test_collect_sessions_skips_null_and_accepts_iso(fake_hermes_home, monkeypatch):
    conn = sqlite3.connect(f"{fake_hermes_home}/state.db")
    conn.execute(
        "INSERT INTO sessions (id, source, title, started_at, message_count,"
        " tool_call_count, input_tokens, output_tokens) VALUES (?,?,?,?,?,?,?,?)",
        ("sess-null", "cli", "null start", None, 7, 0, 0, 0),
    )
    conn.execute(
        "INSERT INTO sessions (id, source, title, started_at, message_count,"
        " tool_call_count, input_tokens, output_tokens) VALUES (?,?,?,?,?,?,?,?)",
        ("sess-iso", "cli", "iso start", "2026-01-02T03:04:05", 3, 0, 0, 0),
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("HERMES_HOME", fake_hermes_home)

    from hermes_hud.collectors.sessions import collect_sessions

    state = collect_sessions()
    ids = {s.id for s in state.sessions}
    assert "sess-null" not in ids, "NULL started_at must be skipped, not dated 1970"
    assert "sess-iso" in ids, "ISO-string started_at must be parsed, not dropped"
    iso = next(s for s in state.sessions if s.id == "sess-iso")
    assert iso.started_at == datetime(2026, 1, 2, 3, 4, 5)
    assert not any(s.started_at.year == 1970 for s in state.sessions)
    assert all(ds.date for ds in state.daily_stats), "empty date buckets must be skipped"
