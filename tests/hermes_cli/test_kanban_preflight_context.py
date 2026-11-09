"""Tests for the kanban-admission preflight section (``_ctx_preflight``).

ACP workers' LLM calls run inside the Devin process and never reach the
gateway's ``agent:start`` emit, so the kanban lane's preflight coverage is
context injection: at worker admission (``include_preflight=True`` —
``gateway/session_kanban.build_kanban_policy`` only), the matcher runs on the
task's ``title + body`` (the same ``goal_text`` the policy carries), writes the
lane's telemetry row, and the rendered context gains a
``## Preflight skill matches`` section. Display callers keep the flag off:
identical renders, no matcher runs. Any matcher failure is a silent no-op —
admission is never blocked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def task_id(kanban_home):
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn, title="Add dark-lane preflight coverage",
            body="Emit agent:start on desktop, subagent, cron; inject matches at kanban admission.")
    finally:
        conn.close()
    return tid


def _conn():
    return kbc.connect()


class TestFlagGating:
    def test_default_render_has_no_preflight_section_and_never_calls_matcher(
            self, kanban_home, task_id, monkeypatch):
        from agent import preflight_inbound as pf
        monkeypatch.setattr(pf, "match_markdown_for_admission", lambda *a, **k: 1 / 0)
        conn = _conn()
        try:
            ctx = kb.build_worker_context(conn, task_id)
        finally:
            conn.close()
        assert "## Preflight skill matches" not in ctx
        assert "Add dark-lane preflight coverage" in ctx

    def test_include_preflight_injects_section(self, kanban_home, task_id, monkeypatch):
        from agent import preflight_inbound as pf
        seen = {}

        def fake_match(inbound_text, *, session_id="", platform="", source="hook"):
            seen.update(inbound_text=inbound_text, session_id=session_id,
                        platform=platform, source=source)
            return "- 0.50 alpha-skill: does the thing\n- 0.40 beta-skill: does the other"

        monkeypatch.setattr(pf, "match_markdown_for_admission", fake_match)
        conn = _conn()
        try:
            ctx = kb.build_worker_context(conn, task_id, include_preflight=True)
        finally:
            conn.close()

        assert "## Preflight skill matches" in ctx
        assert "alpha-skill" in ctx
        # Same goal_text the worker's session policy carries (title + body).
        assert "Add dark-lane preflight coverage" in seen["inbound_text"]
        assert "Emit agent:start on desktop" in seen["inbound_text"]
        # Telemetry row provenance the coverage audit joins on.
        assert seen["session_id"] == f"kanban:{task_id}"
        assert seen["platform"] == "kanban"

    def test_matcher_unavailable_admission_renders_without_section(
            self, kanban_home, task_id, monkeypatch):
        from agent import preflight_inbound as pf
        monkeypatch.setattr(pf, "match_markdown_for_admission", lambda *a, **k: None)
        conn = _conn()
        try:
            ctx = kb.build_worker_context(conn, task_id, include_preflight=True)
        finally:
            conn.close()
        assert "## Preflight skill matches" not in ctx
        # The rest of the admission context renders unharmed.
        assert "Add dark-lane preflight coverage" in ctx

    def test_matcher_exception_never_blocks_admission(
            self, kanban_home, task_id, monkeypatch):
        from agent import preflight_inbound as pf

        def boom(*a, **k):
            raise RuntimeError("matcher exploded")

        monkeypatch.setattr(pf, "match_markdown_for_admission", boom)
        conn = _conn()
        try:
            ctx = kb.build_worker_context(conn, task_id, include_preflight=True)
        finally:
            conn.close()
        assert "Add dark-lane preflight coverage" in ctx


class TestSectionOrdering:
    def test_preflight_section_lands_before_attachments(
            self, kanban_home, task_id, monkeypatch):
        from agent import preflight_inbound as pf
        monkeypatch.setattr(
            pf, "match_markdown_for_admission", lambda *a, **k: "- 0.50 x: y")
        conn = _conn()
        try:
            dest_dir = kb.task_attachments_dir(task_id)
            dest_dir.mkdir(parents=True, exist_ok=True)
            blob = dest_dir / "doc.pdf"
            blob.write_bytes(b"data")
            kb.add_attachment(conn, task_id, filename="doc.pdf",
                              stored_path=str(blob.resolve()),
                              content_type="application/pdf", size=4)
            ctx = kb.build_worker_context(conn, task_id, include_preflight=True)
        finally:
            conn.close()
        assert ctx.index("## Preflight skill matches") < ctx.index("## Attachments")
