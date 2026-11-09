"""delegate_task child spawns emit turn-inbound preflight coverage.

A delegated child's first turn (``_ChildRun.await_child`` →
``child.run_conversation`` inside ``child_profile_scope``) never traverses
the gateway's ``agent:start`` emit — the parent's covered turn is a different
message. The shared helper covers the spawn's goal message under the child's
own profile scope; a failure must never break the spawn.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import delegate_task


_GOAL = "inspect the kanban dispatcher's reclaim path and report what you find"


def _mock_child():
    child = MagicMock()
    child.session_id = "child-sid-1"
    child._delegate_profile_home = None  # no profile route -> parent scope
    child.run_conversation.return_value = {
        "final_response": "done", "completed": True, "api_calls": 1}
    return child


def test_child_spawn_emits_turn_inbound_in_child_scope(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "agent.preflight_inbound.emit_turn_inbound",
        lambda *a, **k: calls.append((a, k)))
    parent = _make_mock_parent(depth=0)
    child = _mock_child()

    with patch("run_agent.AIAgent", return_value=child):
        delegate_task(goal=_GOAL, parent_agent=parent)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (_GOAL,)
    assert kwargs["platform"] == "subagent"
    assert kwargs["session_id"] == "child-sid-1"


def test_emit_failure_never_breaks_the_spawn(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("matcher exploded")

    monkeypatch.setattr("agent.preflight_inbound.emit_turn_inbound", boom)
    parent = _make_mock_parent(depth=0)
    child = _mock_child()

    with patch("run_agent.AIAgent", return_value=child):
        delegate_task(goal=_GOAL, parent_agent=parent)

    child.run_conversation.assert_called_once()
