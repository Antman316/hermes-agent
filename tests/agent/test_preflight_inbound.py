"""Tests for agent/preflight_inbound.py — the shared turn-inbound coverage helper.

The four dark lanes (desktop TUI prompt, delegate_task child spawn, cron fire,
kanban worker admission) never traverse ``gateway/run_turn.py``'s ``agent:start``
emit, so the installed preflight-match hook never runs for them. This helper
calls the same matcher under the same 8 s subprocess bound instead. Contract:
fail-open on trivial input, missing matcher, timeout and any error; identical
(platform, session_id, message) replays inside one process dedup to one row.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from agent import preflight_inbound as pf


@pytest.fixture(autouse=True)
def _reset_dedup():
    pf._RECENT.clear()
    yield
    pf._RECENT.clear()


@pytest.fixture
def matcher_home(tmp_path, monkeypatch):
    """An isolated HERMES_HOME whose scripts/preflight/ holds a stub matcher.

    The stub appends one JSON row per invocation to ``count_log`` (an
    out-of-band invocation counter) and writes a fixed sentinel to ``--out``,
    mirroring the real preflight_match.py's observable contract.
    """
    home = tmp_path / "hermes_home"
    (home / "scripts" / "preflight").mkdir(parents=True)
    count_log = tmp_path / "matcher_calls.jsonl"
    stub = (home / "scripts" / "preflight" / "preflight_match.py")
    stub.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        f"COUNT_LOG = Path({str(count_log)!r})\n"
        "def _flag(name):\n"
        "    a = sys.argv[1:]\n"
        "    return a[a.index(name) + 1] if name in a else ''\n"
        "q = Path(_flag('--query-file')).read_text(encoding='utf-8').strip()\n"
        "row = {'q': q, 'session_id': _flag('--session-id'),\n"
        "       'platform': _flag('--platform'), 'source': _flag('--source'),\n"
        "       'truncated': '--truncated' in sys.argv}\n"
        "with COUNT_LOG.open('a', encoding='utf-8') as fh:\n"
        "    fh.write(json.dumps(row) + '\\n')\n"
        "out = _flag('--out')\n"
        "if out:\n"
        "    Path(out).write_text('<!-- preflight sentinel, auto-generated test -->\\n'\n"
        "        '- 0.50 alpha: first match\\n- 0.40 beta: second match\\n',\n"
        "        encoding='utf-8')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home, count_log


def _calls(count_log: Path) -> list[dict]:
    if not count_log.exists():
        return []
    return [json.loads(line) for line in count_log.read_text().splitlines() if line.strip()]


_MSG = "please refactor the kanban dispatcher so retries keep their claims"


class TestTrivial:
    @pytest.mark.parametrize("msg", [
        "", "   ", "short", "/status", "ok", "thanks", "continue", "STOP",
    ])
    def test_trivial_inputs_noop(self, matcher_home, msg):
        home, count_log = matcher_home
        assert pf.emit_turn_inbound(msg, session_id="s", platform="desktop") is False
        assert _calls(count_log) == []
        assert not (home / "tmp" / "preflight_pre_turn.md").exists()

    def test_non_string_message_noop(self, matcher_home):
        _home, count_log = matcher_home
        assert pf.emit_turn_inbound([{"type": "text", "text": _MSG}]) is False
        assert pf.emit_turn_inbound(None) is False
        assert _calls(count_log) == []


class TestMatcherAbsent:
    def test_no_matcher_installed_noops(self, tmp_path, monkeypatch):
        home = tmp_path / "empty_home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert pf.emit_turn_inbound(_MSG, session_id="s", platform="cron") is False
        assert pf.match_markdown_for_admission(_MSG, session_id="kanban:t", platform="kanban") is None


class TestEmit:
    def test_emit_runs_matcher_and_writes_sentinel(self, matcher_home):
        home, count_log = matcher_home
        assert pf.emit_turn_inbound(_MSG, session_id="agent-sid-1", platform="desktop") is True
        calls = _calls(count_log)
        assert len(calls) == 1
        row = calls[0]
        assert row["q"] == _MSG
        assert row["session_id"] == "agent-sid-1"
        assert row["platform"] == "desktop"
        assert row["source"] == "hook"
        assert row["truncated"] is False
        sentinel = home / "tmp" / "preflight_pre_turn.md"
        assert sentinel.exists()
        assert "alpha" in sentinel.read_text()

    def test_long_message_is_capped_and_flagged(self, matcher_home):
        _home, count_log = matcher_home
        long_msg = "x" * 900
        assert pf.emit_turn_inbound(long_msg, session_id="s", platform="cron") is True
        calls = _calls(count_log)
        assert len(calls[0]["q"]) == 500
        assert calls[0]["truncated"] is True


class TestDedup:
    def test_same_message_session_platform_dedupes(self, matcher_home):
        _home, count_log = matcher_home
        assert pf.emit_turn_inbound(_MSG, session_id="s", platform="desktop") is True
        assert pf.emit_turn_inbound(_MSG, session_id="s", platform="desktop") is False
        assert len(_calls(count_log)) == 1

    def test_dedup_disabled_reruns(self, matcher_home):
        _home, count_log = matcher_home
        assert pf.emit_turn_inbound(_MSG, session_id="s", platform="desktop", dedup=False) is True
        assert pf.emit_turn_inbound(_MSG, session_id="s", platform="desktop", dedup=False) is True
        assert len(_calls(count_log)) == 2

    def test_different_session_not_deduped(self, matcher_home):
        _home, count_log = matcher_home
        assert pf.emit_turn_inbound(_MSG, session_id="s", platform="desktop") is True
        assert pf.emit_turn_inbound(_MSG, session_id="other-session", platform="desktop") is True
        assert len(_calls(count_log)) == 2

    def test_different_platform_not_deduped(self, matcher_home):
        _home, count_log = matcher_home
        assert pf.emit_turn_inbound(_MSG, session_id="s", platform="desktop") is True
        assert pf.emit_turn_inbound(_MSG, session_id="s", platform="subagent") is True
        assert len(_calls(count_log)) == 2

    def test_different_message_not_deduped(self, matcher_home):
        _home, count_log = matcher_home
        assert pf.emit_turn_inbound(_MSG, session_id="s", platform="desktop") is True
        assert pf.emit_turn_inbound(
            "a different message entirely, still long enough",
            session_id="s", platform="desktop") is True
        assert len(_calls(count_log)) == 2

    def test_dedup_concurrent_emits_single_row(self, matcher_home):
        _home, count_log = matcher_home
        results = []
        threads = [threading.Thread(
            target=lambda: results.append(
                pf.emit_turn_inbound(_MSG, session_id="s", platform="desktop")))
            for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        # The 8 s subprocess bound serializes emits anyway; the contract is one row.
        assert results.count(True) == 1
        assert len(_calls(count_log)) == 1

    def test_dedup_failure_degrades_to_emit_not_turn_error(self, matcher_home, monkeypatch):
        _home, count_log = matcher_home
        monkeypatch.setattr(pf, "_dedup_hit", lambda *a: 1 / 0)
        # ZeroDivisionError inside dedup must degrade to emitting, never propagate.
        assert pf.emit_turn_inbound(_MSG, session_id="s", platform="desktop") is True
        assert len(_calls(count_log)) == 1


class TestFailOpen:
    def test_matcher_nonzero_exit_noops(self, matcher_home):
        home, _count_log = matcher_home
        stub = home / "scripts" / "preflight" / "preflight_match.py"
        stub.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
        assert pf.emit_turn_inbound(_MSG, session_id="s", platform="cron") is False

    def test_matcher_timeout_noops(self, matcher_home, monkeypatch):
        home, _count_log = matcher_home
        stub = home / "scripts" / "preflight" / "preflight_match.py"
        stub.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
        monkeypatch.setattr(pf, "_TIMEOUT_SECS", 0.5)
        assert pf.emit_turn_inbound(_MSG, session_id="s", platform="desktop") is False
        assert pf.match_markdown_for_admission(_MSG) is None

    def test_matcher_crash_noops(self, matcher_home):
        home, _count_log = matcher_home
        stub = home / "scripts" / "preflight" / "preflight_match.py"
        stub.write_text("raise SystemExit('boom')\n", encoding="utf-8")
        assert pf.emit_turn_inbound(_MSG, session_id="s", platform="desktop") is False


class TestAdmissionMarkdown:
    def test_admission_returns_body_without_generator_comment(self, matcher_home):
        _home, count_log = matcher_home
        body = pf.match_markdown_for_admission(
            _MSG, session_id="kanban:t_abc", platform="kanban")
        assert body is not None
        assert "<!--" not in body
        assert "alpha: first match" in body
        row = _calls(count_log)[0]
        # The admission run still writes the lane's telemetry row.
        assert row["session_id"] == "kanban:t_abc"
        assert row["platform"] == "kanban"
        assert row["source"] == "hook"

    def test_admission_does_not_clobber_the_shared_sentinel(self, matcher_home):
        home, _count_log = matcher_home
        sentinel = home / "tmp" / "preflight_pre_turn.md"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text("LIVE TURN SENTINEL — do not overwrite", encoding="utf-8")
        assert pf.match_markdown_for_admission(_MSG, session_id="kanban:t", platform="kanban")
        assert sentinel.read_text() == "LIVE TURN SENTINEL — do not overwrite"

    def test_admission_trivial_noops(self, matcher_home):
        _home, count_log = matcher_home
        assert pf.match_markdown_for_admission("ok", session_id="kanban:t") is None
        assert _calls(count_log) == []
