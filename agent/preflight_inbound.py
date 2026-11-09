"""Shared turn-inbound preflight coverage for lanes the gateway ``agent:start`` emit never reaches.

Mirrors the installed preflight-match hook (``hooks/preflight-match/handler.py`` under the Ezra
profile) one-for-one: a non-trivial inbound message runs the ACTIVE profile's
``scripts/preflight/preflight_match.py --source hook --top 5`` in a bounded subprocess, which
appends a telemetry row to ``state/preflight_log.jsonl`` and writes the ranked sentinel
``tmp/preflight_pre_turn.md``. The matcher is a profile resource, resolved per call under
``get_hermes_home()`` — a profile without it installed no-ops, the same outcome as a profile
without the hook dir.

Fail-open is contractual (trivial message, missing matcher, timeout, any error -> no-op), and
the same-process ``(platform, session_id, message-hash)`` dedup keeps a replayed turn — crash
auto-continue, queued drain — from writing a second row for the same message.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

_HOOK_MSG_CAP = 500   # agent:start's message cap: the hook matches the same prefix
_TIMEOUT_SECS = 8     # installed hook's subprocess bound
_TOP = 5
_DEDUP_TTL_S = 300.0  # same-process replay window for an identical (lane, session, message)
_DEDUP_MAX = 512

_RECENT: OrderedDict[tuple, float] = OrderedDict()
# Turn paths call this from many threads (TUI turns, delegate workers, cron fires);
# OrderedDict mutation is not thread-safe, and a racy dedup must never escape as a
# turn error.
_RECENT_LOCK = threading.Lock()


def _trivial(message: str) -> bool:
    m = message.strip()
    if len(m) < 30:
        return True
    if m.startswith("/") and " " not in m:
        return True
    if m.lower() in {"ok", "thanks", "thank you", "yes", "no", "go", "continue", "stop"}:
        return True
    return False


def _profile_matcher() -> Optional[Path]:
    try:
        from hermes_constants import get_hermes_home
        path = Path(get_hermes_home()) / "scripts" / "preflight" / "preflight_match.py"
    except Exception:
        return None
    return path if path.is_file() else None


def _sentinel_path() -> Path:
    from hermes_constants import get_hermes_home
    return Path(get_hermes_home()) / "tmp" / "preflight_pre_turn.md"


def _dedup_hit(platform: str, session_id: str, msg_hash: str) -> bool:
    """Record ``(platform, session_id, msg_hash)``; True when the same key fired within the TTL."""
    now = time.monotonic()
    with _RECENT_LOCK:
        while _RECENT and (len(_RECENT) > _DEDUP_MAX or now - next(iter(_RECENT.values())) > _DEDUP_TTL_S):
            _RECENT.popitem(last=False)
        key = (platform, session_id, msg_hash)
        ts = _RECENT.get(key)
        if ts is not None and now - ts <= _DEDUP_TTL_S:
            return True
        _RECENT[key] = now
        _RECENT.move_to_end(key)
        return False


def _matcher_python() -> str:
    """Interpreter for the matcher subprocess: the repo venv the installed hook hardcodes
    (``venv/bin/python`` beside this package's repo root), else the caller's own. The venv
    preference keeps the helper's matcher runs interpreter-identical to the covered lanes'
    even when the calling service runs on another runtime (tools python, a different venv)."""
    venv_py = Path(__file__).resolve().parents[1] / "venv" / "bin" / "python"
    return str(venv_py) if venv_py.is_file() else sys.executable


def _run_matcher(matcher: Path, message: str, session_id: str, platform: str, source: str,
                 out_path: Path) -> bool:
    """One bounded matcher subprocess; True only on rc==0 (telemetry row + sentinel written).

    Command and env mirror the installed hook's ``_run``: same flags, same 500-char truncation
    marker, same real-home repair so telemetry lands on the profile's own log under sandboxed
    HOME."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, dir=str(out_path.parent), encoding="utf-8"
    ) as fh:
        fh.write(message)
        qfile = fh.name
    try:
        cmd = [_matcher_python(), str(matcher), "--query-file", qfile, "--top", str(_TOP),
               "--source", source, "--out", str(out_path)]
        if session_id:
            cmd += ["--session-id", session_id]
        if platform:
            cmd += ["--platform", platform]
        if len(message) >= _HOOK_MSG_CAP:
            cmd.append("--truncated")
        from hermes_constants import get_real_home
        real_home = get_real_home()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_TIMEOUT_SECS, check=False,
            env={**os.environ, "HOME": real_home, "HERMES_REAL_HOME": real_home})
        return proc.returncode == 0
    finally:
        try:
            os.unlink(qfile)
        except OSError:
            pass


def emit_turn_inbound(message, *, session_id: str = "", platform: str = "",
                      source: str = "hook", dedup: bool = True) -> bool:
    """Preflight-cover one inbound turn message (telemetry row + sentinel). Never raises.

    ``message`` mirrors the hook's ``context['message']``: the raw inbound text. Non-str
    (multimodal content lists) is a no-op — the matcher scores text only.
    """
    if not isinstance(message, str) or _trivial(message):
        return False
    matcher = _profile_matcher()
    if matcher is None:
        return False
    capped = message[:_HOOK_MSG_CAP]
    if dedup:
        try:
            if _dedup_hit(
                    platform, session_id, hashlib.sha256(capped.encode("utf-8")).hexdigest()[:12]):
                return False
        except Exception:
            # A dedup failure degrades to emitting again — an occasional second row is far
            # safer than a turn error on the coverage path.
            pass
    try:
        return _run_matcher(matcher, capped, session_id, platform, source, _sentinel_path())
    except Exception:
        return False


def match_markdown_for_admission(inbound_text, *, session_id: str = "", platform: str = "",
                                 source: str = "hook") -> Optional[str]:
    """Sentinel-format matcher output for injection into a worker's admission context.

    Writes the lane's telemetry row as the matcher run's side effect, but renders to a scratch
    ``--out`` (never the shared sentinel a live turn may be reading) and returns its content
    minus the generator comment; None on any failure so admission is never blocked.
    """
    if not isinstance(inbound_text, str) or _trivial(inbound_text):
        return None
    matcher = _profile_matcher()
    if matcher is None:
        return None
    capped = inbound_text[:_HOOK_MSG_CAP]
    try:
        from hermes_constants import get_hermes_home
        tmp_dir = Path(get_hermes_home()) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "r", suffix=".md", dir=str(tmp_dir), delete=False, encoding="utf-8") as out_f:
            out_path = Path(out_f.name)
    except Exception:
        return None
    try:
        if not _run_matcher(matcher, capped, session_id, platform, source, out_path):
            return None
        body = out_path.read_text(encoding="utf-8-sig").strip()
        lines = [line for line in body.splitlines() if not line.startswith("<!--")]
        return "\n".join(lines).strip() or None
    except Exception:
        return None
    finally:
        try:
            out_path.unlink()
        except OSError:
            pass
