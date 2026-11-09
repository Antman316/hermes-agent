"""Desktop/TUI prompt turns emit turn-inbound preflight coverage.

``_invoke_agent`` (``tui_gateway/prompt_turn.py``) is the single prompt path the
desktop and TUI share; it never traverses ``gateway/run_turn.py``'s
``agent:start`` emit, so the shared ``agent.preflight_inbound`` helper covers
the turn instead — same matcher, same 8 s bound, fail-open by contract (the
call site also suppresses every error, import included).
"""

from __future__ import annotations

import types

from tui_gateway import server
from tests.tui_gateway.test_auto_continue import _session, marker_home, turn_env


_PROMPT = "please refactor the kanban dispatcher so retries keep their claims"


def _agent(**extra):
    return types.SimpleNamespace(
        session_id="agent-sid-1",
        run_conversation=lambda *a, **k: {"final_response": "done"},
        clear_interrupt=lambda: None,
        **extra)


def test_turn_emits_turn_inbound_with_raw_prompt(turn_env, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "agent.preflight_inbound.emit_turn_inbound",
        lambda *a, **k: calls.append((a, k)))
    session = _session(agent=_agent(platform="desktop"), running=True)

    server._run_prompt_submit("rid", "ui-sid", session, _PROMPT)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (_PROMPT,)
    assert kwargs["session_id"] == "agent-sid-1"
    assert kwargs["platform"] == "desktop"


def test_turn_platform_falls_back_to_desktop_then_tui(turn_env, monkeypatch):
    """The emit carries the agent's own platform (``desktop`` chat panel or
    ``tui`` embedded terminal) so rows join the session's ``source`` column."""
    calls = []
    monkeypatch.setattr(
        "agent.preflight_inbound.emit_turn_inbound",
        lambda *a, **k: calls.append((a, k)))

    session = _session(agent=_agent(), running=True)  # no platform attr -> "desktop"
    server._run_prompt_submit("rid", "ui-sid", session, _PROMPT)
    assert calls[0][1]["platform"] == "desktop"

    calls.clear()
    session = _session(agent=_agent(platform="tui"), running=True)
    server._run_prompt_submit("rid", "ui-sid", session, _PROMPT)
    assert calls[0][1]["platform"] == "tui"


def test_emit_failure_never_breaks_the_turn(turn_env, monkeypatch):
    """A raising coverage path (import skew, matcher bug) must not fail the
    user's turn — the suppress at the emit site is part of the contract."""
    ran = []

    def boom(*a, **k):
        raise RuntimeError("matcher exploded")

    monkeypatch.setattr("agent.preflight_inbound.emit_turn_inbound", boom)
    agent = _agent(platform="desktop")
    agent.run_conversation = lambda *a, **k: ran.append(1) or {"final_response": "done"}
    session = _session(agent=agent, running=True)

    assert server._run_prompt_submit("rid", "ui-sid", session, _PROMPT)
    assert ran == [1]
