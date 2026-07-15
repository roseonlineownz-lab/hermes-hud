"""Regression tests for the three high-severity fixes.

1. `hermes-hud --snapshot` must persist a snapshot to snapshots.jsonl.
2. collect_corrections must mine session corrections from state.db
   (the old REGEXP query aborted the whole extraction).
3. Panels must escape user-controlled strings before rendering them as
   Textual markup — `[/...]` in user data raised MarkupError (app crash)
   and `[word]` was silently swallowed (display data loss).
"""

import sqlite3
import sys
import time
from datetime import datetime

import pytest
from textual.content import Content


# ── Fix 1: --snapshot persists ──────────────────────────────────────────

def test_snapshot_flag_persists(env_override, monkeypatch, tmp_path):
    import hermes_hud.snapshot as snapshot_mod
    import hermes_hud.hud as hud_mod

    snap_dir = tmp_path / ".hud"
    monkeypatch.setattr(snapshot_mod, "SNAPSHOT_DIR", str(snap_dir))
    monkeypatch.setattr(sys, "argv", ["hermes-hud", "--snapshot"])

    hud_mod.main()

    snap_file = snap_dir / "snapshots.jsonl"
    assert snap_file.exists(), "--snapshot must write snapshots.jsonl"
    assert len(snap_file.read_text().strip().splitlines()) == 1


# ── Fix 2: session corrections are mined ────────────────────────────────

def test_session_corrections_mined(fake_hermes_home, monkeypatch):
    # The fixture's messages table uses created_at; the collector queries
    # timestamp — rebuild messages with the real column name and insert a
    # user message that matches a correction keyword.
    db = f"{fake_hermes_home}/state.db"
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE messages")
    conn.execute(
        "CREATE TABLE messages (id TEXT PRIMARY KEY, session_id TEXT,"
        " role TEXT, content TEXT, tool_calls TEXT, timestamp REAL)"
    )
    conn.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
        ("m-corr", "sess-0", "user",
         "no, that's wrong - the config lives in /etc, verify before suggesting",
         None, time.time()),
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("HERMES_HOME", fake_hermes_home)

    from hermes_hud.collectors.corrections import collect_corrections

    corrections = collect_corrections()
    assert corrections.by_source().get("session", 0) > 0, (
        "user messages with correction keywords must surface as session corrections"
    )


# ── Fix 3: hostile user data renders literally ──────────────────────────

HOSTILE_CLOSE = "[/var/log]"   # raised MarkupError when unescaped
HOSTILE_TAG = "[wip]"          # silently swallowed when unescaped


def _render_lines(module, panel_factory):
    """Capture every markup string a panel passes to Static during compose."""
    recorded = []

    class Recorder:
        def __init__(self, content="", **kw):
            recorded.append(str(content))

    real = module.Static
    module.Static = Recorder
    try:
        list(panel_factory().compose())
    finally:
        module.Static = real
    return recorded


def _assert_renders(module, panel_factory, *needles):
    plain = []
    for line in _render_lines(module, panel_factory):
        plain.append(Content.from_markup(line).plain)  # raises if unescaped [/...]
    joined = "\n".join(plain)
    for needle in needles:
        assert needle in joined, f"{needle!r} lost from rendered output"


def test_cron_panel_escapes_user_data():
    import hermes_hud.widgets.cron_panel as mod
    from hermes_hud.collectors.cron import CronJob, CronState

    job = CronJob(
        id="j1", name=f"{HOSTILE_TAG} cleanup",
        prompt=f"check {HOSTILE_CLOSE} for errors",
        schedule_display="every 60m", enabled=True, state="scheduled",
        last_run_at="2026-06-10T01:00:00", last_status="error",
        last_error=f"OSError at {HOSTILE_CLOSE}", paused_reason=None,
    )
    _assert_renders(mod, lambda: mod.CronPanel(CronState(jobs=[job])),
                    f"{HOSTILE_TAG} cleanup", HOSTILE_CLOSE)


def test_corrections_panel_escapes_user_data():
    import hermes_hud.widgets.corrections_panel as mod
    from hermes_hud.collectors.corrections import Correction, CorrectionsState

    cor = Correction(
        timestamp=datetime(2026, 6, 1), source="session", summary="s",
        detail=f"you said {HOSTILE_CLOSE} holds {HOSTILE_TAG} configs",
        session_title=f"{HOSTILE_TAG} fix nginx", severity="critical",
    )
    _assert_renders(mod, lambda: mod.CorrectionsPanel(CorrectionsState(corrections=[cor])),
                    HOSTILE_CLOSE, f"{HOSTILE_TAG} configs", f"{HOSTILE_TAG} fix nginx")


def test_timeline_panel_escapes_user_data():
    import hermes_hud.widgets.timeline_panel as mod
    from hermes_hud.models import TimelineEvent

    events = [
        TimelineEvent(timestamp=datetime(2026, 6, 1), event_type="milestone",
                      title=f"{HOSTILE_TAG} first session",
                      detail=f"started {HOSTILE_CLOSE} via cli"),
        TimelineEvent(timestamp=datetime(2026, 6, 2), event_type="session",
                      title=f"{HOSTILE_TAG} daily session"),
    ]
    _assert_renders(mod, lambda: mod.TimelinePanel(events),
                    f"{HOSTILE_TAG} first session", HOSTILE_CLOSE,
                    f"{HOSTILE_TAG} daily session")


def test_projects_panel_escapes_user_data():
    import hermes_hud.widgets.projects_panel as mod
    from hermes_hud.collectors.projects import ProjectInfo, ProjectsState

    proj = ProjectInfo(
        name="proj[1]", path="/x", is_git=True, branch=f"feat/{HOSTILE_TAG}",
        last_commit_msg=f"[ci skip] tweak {HOSTILE_CLOSE} route",
        last_commit_ago="2 hours ago", last_commit_ts=time.time(),
        dirty_files=1, total_commits=3, languages=["Python"],
    )
    _assert_renders(mod, lambda: mod.ProjectsPanel(ProjectsState(projects=[proj], projects_dir="/x")),
                    "proj[1]", f"feat/{HOSTILE_TAG}", "[ci skip]", HOSTILE_CLOSE)


def test_agents_panel_escapes_user_data():
    import hermes_hud.widgets.agents_panel as mod
    from hermes_hud.collectors.agents import AgentProcess, AgentsState, RecentSession
    from hermes_hud.collectors.cron import CronState

    state = AgentsState(
        processes=[AgentProcess(name="hermes", binary="hermes", running=True,
                                pid=1234, cwd=f"~/projects/{HOSTILE_TAG} thing",
                                tmux_jump_hint="my[session]:0.1")],
        recent_sessions=[RecentSession(session_id="s1", source="cli",
                                       title=f"{HOSTILE_TAG} chat",
                                       started_at=datetime(2026, 6, 1))],
    )
    _assert_renders(mod, lambda: mod.AgentsPanel(state, CronState()),
                    f"~/projects/{HOSTILE_TAG} thing", "my[session]:0.1",
                    f"{HOSTILE_TAG} chat")


def test_skills_overview_profiles_panels_escape_user_data():
    import hermes_hud.widgets.skills_panel as skills_mod
    import hermes_hud.widgets.overview as overview_mod
    import hermes_hud.widgets.profiles_panel as profiles_mod
    from hermes_hud.models import (
        ConfigState, HUDState, ProfileInfo, ProfilesState, SkillInfo, SkillsState,
    )

    skills = SkillsState(skills=[
        SkillInfo(name=f"{HOSTILE_TAG} skill", category="dev[ops]",
                  description="", path="/x", modified_at=datetime(2026, 6, 1)),
    ])
    _assert_renders(skills_mod, lambda: skills_mod.SkillsPanel(skills),
                    f"{HOSTILE_TAG} skill", "dev[ops]")

    hud_state = HUDState(config=ConfigState(model="m", provider="anthropic",
                                            toolsets=["term[inal]"], backend="local"))
    _assert_renders(overview_mod, lambda: overview_mod.OverviewPanel(hud_state),
                    "term[inal]")

    profiles = ProfilesState(profiles=[
        ProfileInfo(name="default", is_default=True, provider="anthropic",
                    soul_summary=f"I guard {HOSTILE_CLOSE} and {HOSTILE_TAG} dreams"),
    ])
    _assert_renders(profiles_mod, lambda: profiles_mod.ProfilesPanel(profiles),
                    HOSTILE_CLOSE, f"{HOSTILE_TAG} dreams")
