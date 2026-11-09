"""Cron fires emit turn-inbound preflight coverage.

``cron.scheduler.run_job`` is the single agent-run path for both fire modes:
a standalone tick delegates to ``run_canonical_job`` → the owner gateway's
``session_cron.execute`` → ``run_job`` again with the cron identity bound, and
``no_agent``/in-process fires run it directly. Neither ever traverses
``gateway/run_turn.py``'s ``agent:start`` emit, so the shared
``agent.preflight_inbound`` helper covers the job prompt — same matcher,
same 8 s bound, fail-open (the call site also suppresses every error).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from tests.cron.test_scheduler import _run_owned_job


_JOB = {
    "id": "preflight-cron-job",
    "name": "preflight-cron",
    "prompt": "summarize today's kanban board state and flag blocked tasks",
    "deliver": "local",
}


class _FakeAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run_conversation(self, *args, **kwargs):
        return {"final_response": "ok"}


def _patches(tmp_path, fake_db):
    return [
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._preflight_job_config", return_value=None),
        patch("hermes_state_registry.acquire", return_value=fake_db),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "***",
                "base_url": "https://example.invalid/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
            },
        ),
        patch("run_agent.AIAgent", _FakeAgent),
    ]


def _apply(patches):
    import contextlib
    stack = contextlib.ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


def test_cron_fire_emits_turn_inbound(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "agent.preflight_inbound.emit_turn_inbound",
        lambda *a, **k: calls.append((a, k)))
    fake_db = MagicMock()

    with _apply(_patches(tmp_path, fake_db)):
        success, _doc, _resp, _err = _run_owned_job(dict(_JOB), tmp_path, fake_db)

    assert success is True
    assert len(calls) == 1
    args, kwargs = calls[0]
    # The emit covers the assembled prompt the agent actually receives
    # (cron hint + run-time line + the job's stored prompt).
    assert isinstance(args[0], str) and _JOB["prompt"] in args[0]
    assert kwargs["platform"] == "cron"
    # The fire's owner session id — the harness binds "canonical-cron-session".
    assert kwargs["session_id"] == "canonical-cron-session"


def test_emit_failure_never_fails_the_fire(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("matcher exploded")

    monkeypatch.setattr("agent.preflight_inbound.emit_turn_inbound", boom)
    fake_db = MagicMock()

    with _apply(_patches(tmp_path, fake_db)):
        success, _doc, resp, _err = _run_owned_job(dict(_JOB), tmp_path, fake_db)

    assert success is True
    assert resp == "ok"
