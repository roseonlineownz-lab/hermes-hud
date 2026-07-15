"""Adversarial tests: every collector must survive malformed data sources.

Each test builds a hostile ~/.hermes tree (binary garbage, wrong types,
directories where files belong, corrupt databases, ancient schemas) and
asserts the collector degrades gracefully instead of raising.
"""

import json
import os
import sqlite3

import pytest

from hermes_hud.models import (
    ConfigState,
    MemoryState,
    PatternsState,
    SessionsState,
    SkillsState,
)


INVALID_UTF8 = b"\xff\xfe\x80\x81 not utf-8 \xc3\x28"


def _make_db(path, schema_sql, rows=()):
    """Create a sqlite db at path with the given schema and rows."""
    conn = sqlite3.connect(str(path))
    for stmt in schema_sql:
        conn.execute(stmt)
    for table_sql, params in rows:
        conn.execute(table_sql, params)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------

class TestMemoryAdversarial:
    def _collect(self, hermes):
        from hermes_hud.collectors.memory import collect_memory
        return collect_memory(str(hermes))

    def test_invalid_utf8(self, tmp_path):
        mem = tmp_path / "memories"
        mem.mkdir()
        (mem / "MEMORY.md").write_bytes(INVALID_UTF8)
        (mem / "USER.md").write_bytes(INVALID_UTF8)
        memory, user = self._collect(tmp_path)
        assert isinstance(memory, MemoryState)
        assert isinstance(user, MemoryState)
        assert memory.entry_count == 0

    def test_memory_file_is_directory(self, tmp_path):
        (tmp_path / "memories" / "MEMORY.md").mkdir(parents=True)
        memory, user = self._collect(tmp_path)
        assert memory.entry_count == 0

    def test_only_delimiters(self, tmp_path):
        mem = tmp_path / "memories"
        mem.mkdir()
        (mem / "MEMORY.md").write_text("§§§\n§  §")
        memory, _ = self._collect(tmp_path)
        assert memory.entry_count == 0

    def test_markup_and_control_chars(self, tmp_path):
        mem = tmp_path / "memories"
        mem.mkdir()
        (mem / "MEMORY.md").write_text(
            "[red]markup injection[/red] \x00\x1b[31m ansi\n§\nnormal entry"
        )
        memory, _ = self._collect(tmp_path)
        assert memory.entry_count == 2

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
    def test_unreadable_file(self, tmp_path):
        mem = tmp_path / "memories"
        mem.mkdir()
        f = mem / "MEMORY.md"
        f.write_text("secret")
        f.chmod(0o000)
        try:
            memory, _ = self._collect(tmp_path)
            assert memory.entry_count == 0
        finally:
            f.chmod(0o644)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class TestConfigAdversarial:
    def _collect(self, hermes):
        from hermes_hud.collectors.config import collect_config
        return collect_config(str(hermes))

    def test_invalid_utf8(self, tmp_path):
        (tmp_path / "config.yaml").write_bytes(INVALID_UTF8)
        assert isinstance(self._collect(tmp_path), ConfigState)

    def test_config_is_directory(self, tmp_path):
        (tmp_path / "config.yaml").mkdir()
        assert isinstance(self._collect(tmp_path), ConfigState)

    @pytest.mark.parametrize("content", [
        "- just\n- a\n- list\n",          # top-level list
        "just a scalar",                   # top-level scalar
        "42",                              # top-level number
        "",                                # empty file
        "model:\ntoolsets:\nagent:\n",     # all-null sections
        "model: [a, b]\n",                 # model as list
        "toolsets: {a: 1}\n",              # toolsets as dict
        "{invalid yaml: [unclosed\n",      # malformed yaml
        "\tmodel: tab-indented\n",         # tabs (yaml error)
    ])
    def test_degenerate_yaml(self, tmp_path, content):
        (tmp_path / "config.yaml").write_text(content)
        config = self._collect(tmp_path)
        assert isinstance(config, ConfigState)
        assert isinstance(config.toolsets, list)

    def test_model_scalar_is_stringified(self, tmp_path):
        (tmp_path / "config.yaml").write_text("model: 123\n")
        config = self._collect(tmp_path)
        assert config.model == "123"


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------

class TestSkillsAdversarial:
    def _collect(self, hermes):
        from hermes_hud.collectors.skills import collect_skills
        return collect_skills(str(hermes))

    def test_binary_skill_md(self, tmp_path):
        d = tmp_path / "skills" / "cat" / "tool"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_bytes(INVALID_UTF8)
        skills = self._collect(tmp_path)
        assert isinstance(skills, SkillsState)
        assert skills.total == 1  # still listed, metadata empty

    def test_skill_md_is_directory(self, tmp_path):
        (tmp_path / "skills" / "cat" / "weird" / "SKILL.md").mkdir(parents=True)
        skills = self._collect(tmp_path)
        assert isinstance(skills, SkillsState)

    def test_skill_md_at_root_is_skipped(self, tmp_path):
        d = tmp_path / "skills"
        d.mkdir()
        (d / "SKILL.md").write_text("---\nname: rootless\n---\n")
        skills = self._collect(tmp_path)
        assert skills.total == 0

    def test_garbage_frontmatter(self, tmp_path):
        d = tmp_path / "skills" / "cat" / "tool"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\n:::: no\nname: : : double\ndescription\n---\nbody [red]markup[/red]\n"
        )
        skills = self._collect(tmp_path)
        assert skills.total == 1


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

class TestSessionsAdversarial:
    def _collect(self, hermes):
        from hermes_hud.collectors.sessions import collect_sessions
        return collect_sessions(str(hermes))

    def test_garbage_db_bytes(self, tmp_path):
        (tmp_path / "state.db").write_bytes(b"definitely not sqlite" * 200)
        sessions = self._collect(tmp_path)
        assert isinstance(sessions, SessionsState)
        assert sessions.total_sessions == 0

    def test_empty_db_no_tables(self, tmp_path):
        _make_db(tmp_path / "state.db", [])
        sessions = self._collect(tmp_path)
        assert sessions.total_sessions == 0

    def test_ancient_schema_missing_columns(self, tmp_path):
        _make_db(tmp_path / "state.db", [
            "CREATE TABLE sessions (id TEXT, started_at REAL)",
            "CREATE TABLE messages (id TEXT, content TEXT)",
        ], [("INSERT INTO sessions VALUES (?, ?)", ("s1", 1700000000.0))])
        sessions = self._collect(tmp_path)
        assert isinstance(sessions, SessionsState)

    def test_hostile_row_values(self, tmp_path):
        """NULLs, huge/negative/garbage timestamps, non-dict tool_calls."""
        _make_db(tmp_path / "state.db", [
            """CREATE TABLE sessions (
                id TEXT, source TEXT, title TEXT, started_at, ended_at,
                message_count, tool_call_count, input_tokens, output_tokens,
                cache_read_tokens, cache_write_tokens, reasoning_tokens,
                estimated_cost_usd, model TEXT, model_config TEXT)""",
            "CREATE TABLE messages (id TEXT, session_id TEXT, role TEXT, content TEXT, tool_calls TEXT, created_at REAL)",
        ], [
            # all NULLs
            ("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
             (None,) * 15),
            # huge timestamp (OverflowError territory)
            ("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
             ("s-huge", "cli", "t", 1e20, 1e20, 1, 1, 0, 0, 0, 0, 0, 0.0, None, None)),
            # negative + garbage string timestamps, garbage model_config
            ("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
             ("s-neg", "cli", "t", -1e15, "not-a-date", 1, 1, 0, 0, 0, 0, 0, 0.0, None, "{broken json")),
            # valid session so we know parsing continues past bad rows
            ("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
             ("s-ok", "cli", "good", 1700000000.0, 1700003600.0, 5, 2, 100, 50, 0, 0, 0, 0.01, "m", None)),
            # tool_calls: valid JSON but wrong shapes
            ("INSERT INTO messages VALUES (?,?,?,?,?,?)",
             ("m1", "s-ok", "assistant", "x", json.dumps([1, "two", None, {"function": "not-a-dict"}, {"no_function": 1}]), 1700000000.0)),
            # tool_calls: broken JSON
            ("INSERT INTO messages VALUES (?,?,?,?,?,?)",
             ("m2", "s-ok", "assistant", "x", "{]", 1700000000.0)),
            # tool_calls: JSON scalar
            ("INSERT INTO messages VALUES (?,?,?,?,?,?)",
             ("m3", "s-ok", "assistant", "x", '"just a string"', 1700000000.0)),
        ])
        sessions = self._collect(tmp_path)
        assert isinstance(sessions, SessionsState)
        ids = [s.id for s in sessions.sessions]
        assert "s-ok" in ids
        assert isinstance(sessions.tool_usage, dict)


# ---------------------------------------------------------------------------
# cron
# ---------------------------------------------------------------------------

class TestCronAdversarial:
    def _collect(self, hermes):
        from hermes_hud.collectors.cron import collect_cron
        return collect_cron(str(hermes))

    @pytest.mark.parametrize("payload", [
        b"not json at all {",
        INVALID_UTF8,
        b'"scalar"',
        b"42",
        b'{"jobs": "not-a-list"}',
        b'{"jobs": {"a": 1}}',
        b'[1, 2, "three"]',                          # bare list of non-dicts
        b'{"jobs": [null, 1, [], "x"]}',              # non-dict jobs
    ])
    def test_degenerate_jobs_json(self, tmp_path, payload):
        cron_dir = tmp_path / "cron"
        cron_dir.mkdir()
        (cron_dir / "jobs.json").write_bytes(payload)
        cron = self._collect(tmp_path)
        assert cron.total == 0

    def test_jobs_json_is_directory(self, tmp_path):
        (tmp_path / "cron" / "jobs.json").mkdir(parents=True)
        assert self._collect(tmp_path).total == 0

    def test_job_with_wrong_field_types(self, tmp_path):
        cron_dir = tmp_path / "cron"
        cron_dir.mkdir()
        (cron_dir / "jobs.json").write_text(json.dumps({"jobs": [{
            "id": None,
            "name": None,
            "prompt": ["a", "list"],
            "schedule": "every day",         # string, not dict
            "schedule_display": None,
            "enabled": "maybe",
            "state": 42,
            "repeat": "3",                   # string, not dict
            "skills": {"a": 1},              # dict, not list
            "last_error": {"nested": True},
        }]}))
        cron = self._collect(tmp_path)
        assert cron.total == 1
        job = cron.jobs[0]
        assert job.name == "unnamed"
        assert job.skills == []
        assert isinstance(job.enabled, bool)
        # state/error with wrong types must not break the aggregate properties
        assert isinstance(cron.active, int)
        assert isinstance(cron.has_errors, bool)


# ---------------------------------------------------------------------------
# corrections
# ---------------------------------------------------------------------------

class TestCorrectionsAdversarial:
    def _collect(self, hermes):
        from hermes_hud.collectors.corrections import collect_corrections
        return collect_corrections(str(hermes))

    def test_no_messages_table(self, tmp_path):
        _make_db(tmp_path / "state.db", ["CREATE TABLE sessions (id TEXT, title TEXT)"])
        corrections = self._collect(tmp_path)
        assert corrections.total == 0

    def test_garbage_db(self, tmp_path):
        (tmp_path / "state.db").write_bytes(b"garbage" * 100)
        assert self._collect(tmp_path).total == 0

    def test_hostile_message_rows(self, tmp_path):
        _make_db(tmp_path / "state.db", [
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT)",
            "CREATE TABLE messages (id TEXT, session_id TEXT, role TEXT, content TEXT, timestamp REAL)",
        ], [
            ("INSERT INTO sessions VALUES (?, ?)", ("s1", "[red]title[/red]")),
            # huge timestamp on a matching keyword row
            ("INSERT INTO messages VALUES (?,?,?,?,?)",
             ("m1", "s1", "user", "that answer is wrong, please verify your work here", 1e20)),
            # NULL content
            ("INSERT INTO messages VALUES (?,?,?,?,?)",
             ("m2", "s1", "user", None, 1700000000.0)),
            # normal correction-like row
            ("INSERT INTO messages VALUES (?,?,?,?,?)",
             ("m3", "s1", "user", "actually that is incorrect, push back on it", 1700000000.0)),
        ])
        corrections = self._collect(tmp_path)
        assert corrections.total >= 1
        assert isinstance(corrections.by_severity(), dict)


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

class TestHealthAdversarial:
    def _collect(self, hermes):
        from hermes_hud.collectors.health import collect_health
        return collect_health(str(hermes))

    def test_binary_dotenv(self, tmp_path):
        (tmp_path / ".env").write_bytes(b"GOOD_API_KEY=x\n" + INVALID_UTF8 + b"=y\nOTHER_TOKEN=\xff\n")
        health = self._collect(tmp_path)
        assert health is not None
        names = [k.name for k in health.keys]
        assert "GOOD_API_KEY" in names

    def test_dotenv_is_directory(self, tmp_path):
        (tmp_path / ".env").mkdir()
        assert self._collect(tmp_path) is not None

    @pytest.mark.parametrize("payload", [
        b"not json",
        INVALID_UTF8,
        b'{"pid": "abc"}',
        b'{"pid": -1}',
        b'{"pid": null}',
        b'[1, 2, 3]',
        b'"scalar"',
    ])
    def test_degenerate_pid_file(self, tmp_path, payload):
        (tmp_path / "gateway.pid").write_bytes(payload)
        health = self._collect(tmp_path)
        gateway = next(s for s in health.services if s.name == "Telegram Gateway")
        assert gateway.running is False

    def test_binary_config(self, tmp_path):
        (tmp_path / "config.yaml").write_bytes(INVALID_UTF8)
        health = self._collect(tmp_path)
        assert health.config_model == ""


# ---------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------

class TestPatternsAdversarial:
    def _collect(self, hermes):
        from hermes_hud.collectors.patterns import collect_patterns
        return collect_patterns(str(hermes))

    def test_garbage_db(self, tmp_path):
        (tmp_path / "state.db").write_bytes(b"junk" * 100)
        patterns = self._collect(tmp_path)
        assert isinstance(patterns, PatternsState)

    def test_missing_columns(self, tmp_path):
        _make_db(tmp_path / "state.db", [
            "CREATE TABLE sessions (id TEXT)",
            "CREATE TABLE messages (id TEXT)",
        ])
        assert isinstance(self._collect(tmp_path), PatternsState)

    def test_hostile_rows(self, tmp_path):
        _make_db(tmp_path / "state.db", [
            "CREATE TABLE sessions (id TEXT, title TEXT, message_count, tool_call_count, started_at)",
            "CREATE TABLE messages (id TEXT, session_id TEXT, role TEXT, content TEXT, tool_calls TEXT, timestamp REAL)",
        ], [
            ("INSERT INTO sessions VALUES (?,?,?,?,?)", (None, None, None, None, None)),
            ("INSERT INTO sessions VALUES (?,?,?,?,?)", ("s1", "fix the bug", "not-a-number", 2, 1e20)),
            ("INSERT INTO sessions VALUES (?,?,?,?,?)", ("s2", "commit changes", 3, 1, 1700000000.0)),
            ("INSERT INTO messages VALUES (?,?,?,?,?,?)",
             ("m1", "s2", "user", "please commit my changes", None, 1700000000.0)),
            ("INSERT INTO messages VALUES (?,?,?,?,?,?)",
             ("m2", "s2", "assistant", "x", '[{"function": null}, "str", 7]', 1700000001.0)),
            ("INSERT INTO messages VALUES (?,?,?,?,?,?)",
             ("m3", "s2", "assistant", "x", "{bad json", 1700000002.0)),
        ])
        patterns = self._collect(tmp_path)
        assert isinstance(patterns, PatternsState)
        assert len(patterns.hourly_activity) == 24


# ---------------------------------------------------------------------------
# projects
# ---------------------------------------------------------------------------

class TestProjectsAdversarial:
    def _collect(self, projects_dir):
        from hermes_hud.collectors.projects import collect_projects
        return collect_projects(str(projects_dir))

    def test_projects_dir_is_a_file(self, tmp_path):
        f = tmp_path / "projects"
        f.write_text("not a dir")
        assert self._collect(f).total == 0

    def test_missing_projects_dir(self, tmp_path):
        assert self._collect(tmp_path / "nope").total == 0

    def test_fake_git_dir(self, tmp_path):
        """A .git directory with no actual repo inside must not crash."""
        proj = tmp_path / "broken-repo"
        (proj / ".git").mkdir(parents=True)
        projects = self._collect(tmp_path)
        assert projects.total == 1
        p = projects.projects[0]
        assert p.is_git is True
        assert isinstance(p.activity_level, str)
        assert isinstance(p.status_label, str)

    def test_git_file_worktree_style(self, tmp_path):
        """Worktrees have a .git *file*, not a directory."""
        proj = tmp_path / "worktree"
        proj.mkdir()
        (proj / ".git").write_text("gitdir: /elsewhere\n")
        projects = self._collect(tmp_path)
        assert projects.projects[0].is_git is False  # treated as non-git, no crash

    def test_broken_symlink_entry(self, tmp_path):
        (tmp_path / "dangling").symlink_to(tmp_path / "does-not-exist")
        assert isinstance(self._collect(tmp_path).total, int)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
    def test_unreadable_project_dir(self, tmp_path):
        proj = tmp_path / "locked"
        proj.mkdir()
        proj.chmod(0o000)
        try:
            projects = self._collect(tmp_path)
            assert projects.total >= 1
        finally:
            proj.chmod(0o755)


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

class TestProfilesAdversarial:
    def _collect_single(self, profile_dir, name="default"):
        from hermes_hud.collectors.profiles import _collect_single_profile
        return _collect_single_profile(profile_dir, name)

    @pytest.mark.parametrize("content", [
        "model: 123\n",
        "model: [a, b]\n",
        "model:\n",
        "model: {provider: x}\n",           # dict without 'default'
        "- top\n- level\n- list\n",
        "compression: nope\n",
        "compression:\n  enabled: 'TRUE'\n",
        "display: just-a-string\n",
        "memory:\n  memory_char_limit: lots\n",
        "model:\n  context_length: 32k\n",
    ])
    def test_degenerate_config(self, tmp_path, content):
        (tmp_path / "config.yaml").write_text(content)
        profile = self._collect_single(tmp_path)
        assert profile.name == "default"
        assert isinstance(profile.context_length, int)
        assert isinstance(profile.memory_max_chars, int)

    def test_binary_everything(self, tmp_path):
        (tmp_path / "config.yaml").write_bytes(INVALID_UTF8)
        (tmp_path / "SOUL.md").write_bytes(INVALID_UTF8)
        (tmp_path / ".env").write_bytes(INVALID_UTF8)
        (tmp_path / "memories").mkdir()
        (tmp_path / "memories" / "MEMORY.md").write_bytes(INVALID_UTF8)
        (tmp_path / "state.db").write_bytes(b"junk" * 50)
        cron = tmp_path / "cron"
        cron.mkdir()
        (cron / "jobs.json").write_bytes(b"{bad")
        profile = self._collect_single(tmp_path)
        assert profile.session_count == 0
        assert profile.memory_entries == 0
        assert profile.cron_job_count == 0

    def test_profiles_dir_contains_files(self, tmp_path, monkeypatch):
        """Stray files inside profiles/ must be skipped."""
        from hermes_hud.collectors.profiles import collect_profiles
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "README.txt").write_text("not a profile")
        (profiles_dir / ".hidden").mkdir()
        state = collect_profiles(str(tmp_path))
        names = [p.name for p in state.profiles]
        assert names == ["default"]

    def test_hostile_base_url(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "model:\n  base_url: 'http://localhost:notaport/v1'\n"
        )
        profile = self._collect_single(tmp_path)
        assert profile.port is None


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------

class TestAgentsAdversarial:
    def test_recent_sessions_hostile_db(self, tmp_path):
        from hermes_hud.collectors.agents import _get_recent_sessions
        _make_db(tmp_path / "state.db", [
            "CREATE TABLE sessions (id TEXT, source TEXT, title TEXT, started_at, ended_at, model TEXT)",
            "CREATE TABLE messages (id TEXT, session_id TEXT, tool_calls TEXT)",
        ], [
            ("INSERT INTO sessions VALUES (?,?,?,?,?,?)", (None, None, None, None, None, None)),
            ("INSERT INTO sessions VALUES (?,?,?,?,?,?)", ("s1", "cli", "t", 1e20, -1e20, "m")),
            ("INSERT INTO sessions VALUES (?,?,?,?,?,?)", ("s2", "cli", "t", "garbage", "junk", "m")),
        ])
        sessions = _get_recent_sessions(str(tmp_path))
        assert isinstance(sessions, list)

    def test_recent_sessions_garbage_db(self, tmp_path):
        from hermes_hud.collectors.agents import _get_recent_sessions
        (tmp_path / "state.db").write_bytes(b"nope" * 100)
        assert _get_recent_sessions(str(tmp_path)) == []

    def test_parse_etime_garbage(self):
        from hermes_hud.collectors.agents import _parse_etime
        for garbage in ("", "abc", "1-2-3", ":::", "-5", "99-", "1:2:3:4"):
            assert isinstance(_parse_etime(garbage), int)


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

class TestParseTimestampAdversarial:
    def test_hostile_values_return_none(self):
        from hermes_hud.collectors.utils import parse_timestamp
        hostile = [
            1e20, -1e20, 2**63, -(2**63),
            float("nan"), float("inf"), float("-inf"),
            "not a date", "", "1e999", "NaN",
            [], {}, object(), b"1700000000",
        ]
        for value in hostile:
            assert parse_timestamp(value) is None, f"expected None for {value!r}"

    def test_valid_values_still_parse(self):
        from hermes_hud.collectors.utils import parse_timestamp
        assert parse_timestamp(1700000000).year == 2023
        assert parse_timestamp(1700000000.5).year == 2023
        assert parse_timestamp("1700000000").year == 2023
        assert parse_timestamp("2023-11-14T22:13:20").year == 2023


# ---------------------------------------------------------------------------
# timeline (pure function over collected state)
# ---------------------------------------------------------------------------

class TestTimelineAdversarial:
    def test_empty_state(self):
        from datetime import datetime
        from hermes_hud.collectors.timeline import build_timeline
        from hermes_hud.models import HUDState, MemoryState, SessionsState, SkillsState, ConfigState

        state = HUDState(
            memory=MemoryState(source="memory"),
            user=MemoryState(source="user"),
            skills=SkillsState(),
            sessions=SessionsState(),
            config=ConfigState(),
            timeline=[],
            collected_at=datetime.now(),
        )
        assert build_timeline(state) == []
