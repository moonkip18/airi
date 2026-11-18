import logging
import os
import sys

import structlog

from app.config import settings


# ---------------------------------------------------------------------------
# ANSI helpers (Windows terminals get VT enabled via os.system("") below)
# ---------------------------------------------------------------------------

_ANSI = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "gray": "\x1b[90m",
    "white": "\x1b[97m",
}


def _c(color: str, text: str) -> str:
    return f"{_ANSI[color]}{text}{_ANSI['reset']}"


def _num(value, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "?"


def _preview(text, max_len: int = 48) -> str:
    if not text:
        return ""
    text = str(text).replace("\r", " ").replace("\n", " / ").strip()
    if len(text) > max_len:
        text = text[: max_len - 1] + "..."
    return text


def _short_time(ts) -> str:
    """ISO timestamp -> HH:MM:SS."""
    if isinstance(ts, str) and len(ts) >= 19:
        return ts[11:19]
    return "?"


def _user(uid) -> str:
    try:
        return f"@u{int(uid)}"
    except (TypeError, ValueError):
        return "@?"


def _chat(cid) -> str:
    try:
        return f"chat {int(cid)}"
    except (TypeError, ValueError):
        return "chat ?"


# ---------------------------------------------------------------------------
# Per-event renderers. Each returns the text AFTER the tag column.
# ---------------------------------------------------------------------------

def _r_in(d) -> str:
    who = d.get("username") or _user(d.get("user_id"))
    bits = [
        _c("cyan", f"{who}:"),
        _c("white", f'"{_preview(d.get("text_preview"), 52)}"'),
    ]
    meta = []
    if d.get("message_type"):
        meta.append(str(d.get("message_type")))
    if d.get("input_language"):
        meta.append(str(d.get("input_language")))
    if d.get("is_direct_mention"):
        meta.append(_c("cyan", "MENTION"))
    if d.get("target") and d.get("target") != "general_group_message":
        meta.append(str(d.get("target")))
    if meta:
        bits.append(_c("dim", "| " + " | ".join(str(m) for m in meta)))
    return " ".join(bits)


def _r_score(d) -> str:
    verdict = (
        _c("green", "participate")
        if d.get("should_participate")
        else _c("gray", "hold back")
    )
    return (
        f"score {_num(d.get('score'), 3)} | att {_num(d.get('attention'))}"
        f" | bored {_num(d.get('boredom'))} | burst {_num(d.get('burst'))} -> {verdict}"
    )


def _r_decide(d) -> str:
    action = str(d.get("action", "?")).upper()
    color = {
        "REPLY": "green",
        "INITIATE": "green",
        "REACT": "yellow",
        "WAIT": "gray",
        "IGNORE": "gray",
        "LEAVE": "red",
    }.get(action, "white")
    bits = [
        _c(color, _c("bold", action)),
        f"conf {_num(d.get('confidence'))}",
        f"att {_num(d.get('attention'))}",
    ]
    if d.get("language"):
        bits.append(f"lang {d.get('language')}")
    reasoning = _preview(d.get("reasoning"), 60)
    if reasoning:
        bits.append(_c("dim", f'"{reasoning}"'))
    return " | ".join(bits)


def _r_out(d) -> str:
    bits = [_c("white", f'"{_preview(d.get("text_preview"), 60)}"')]
    if d.get("delay") is not None:
        bits.append(_c("dim", f"delay {_num(d.get('delay'), 1)}s"))
    if d.get("chat_id") is not None:
        bits.append(_c("dim", "-> " + _chat(d.get("chat_id"))))
    return " ".join(bits)


def _r_react_out(d) -> str:
    target = d.get("message_id") or d.get("conversation_id") or "?"
    return f"{d.get('emoji', '...')} -> msg {target}"


def _r_reaction_seen(d) -> str:
    who = d.get("user_id") and _user(d.get("user_id")) or "?"
    hers = _c("yellow", "hers") if d.get("on_hers") else _c("gray", "others'")
    return f"{who} {d.get('emoji', '?')}" + _c("dim", f" on {hers} msg | {_chat(d.get('chat_id'))}")


def _r_recall(d) -> str:
    return (
        f"{d.get('returned', '?')}/{d.get('candidates', '?')} top {_num(d.get('top_score'))}"
        + _c("dim", f' "{_preview(d.get("query_preview"), 36)}"')
    )


def _r_mem_new(d) -> str:
    return (
        f"{d.get('type', '?')} imp {_num(d.get('importance', 1))}"
        + _c("white", f' "{_preview(d.get("content_preview"), 56)}"')
    )


def _r_mem_superseded(d) -> str:
    return (
        _c("dim", f'"{_preview(d.get("old_content"), 34)}"')
        + " => "
        + _c("white", f'"{_preview(d.get("new_content"), 34)}"')
    )


def _r_state(d) -> str:
    st = d.get("state") or {}
    keys = ("mood", "energy", "curiosity", "amusement", "annoyance", "boredom")
    labels = {"mood": "mood", "energy": "nrg", "curiosity": "cur",
              "amusement": "amu", "annoyance": "ann", "boredom": "brd"}
    bits = [_c("yellow", str(d.get("event_type", "?")))]
    for k in keys:
        if k in st:
            bits.append(f"{labels[k]} {_num(st[k])}")
    return " | ".join(bits)


def _r_rel(d) -> str:
    return (
        f"{_user(d.get('user_id'))} | {d.get('type', '?')}"
        f" | aff {_num(d.get('affection'))} | anno {_num(d.get('annoyance'))}"
        + (f" | jeal {_num(d.get('jealousy'))}" if d.get("jealousy") else "")
    )


def _r_life(d) -> str:
    bits = []
    if d.get("old") or d.get("new"):
        bits.append(f"{d.get('old', '?')} -> {_c('blue', _c('bold', str(d.get('new', '?'))))}")
    if d.get("conversation_id") is not None:
        bits.append(_c("dim", _chat(d.get("conversation_id"))))
    return " ".join(bits) or _c("dim", "lifecycle")


def _r_topic(d) -> str:
    label = str(d.get("event", "topic"))
    if label == "topic_pending":
        return (
            f"{d.get('pending', '?')} {_c('dim', 'pending (still on:')} "
            f"{_c('blue', str(d.get('current', '?')))}{_c('dim', ')')}"
        )
    if label == "topic_set":
        return _c("blue", _c("bold", str(d.get("topic", "?"))))
    return (
        f"{_c('blue', _c('bold', str(d.get('topic', '?'))))}"
        + _c("dim", f" (was: {d.get('current', '?')})")
        if d.get("current")
        else _c("blue", _c("bold", str(d.get("topic", "?"))))
    )


def _r_lang(d) -> str:
    bits = [f"{d.get('input_language', '?')} -> {_c('cyan', str(d.get('output_language', '?')))}"]
    if d.get("allow_persian_snippet"):
        bits.append(_c("cyan", "+snippet"))
    bits.append(_c("dim", str(d.get("reason", ""))))
    return " | ".join(bits)


def _r_timer(d) -> str:
    bits = [_c("yellow", str(d.get("action_type", "?")))]
    if d.get("conversation_id") is not None:
        bits.append(_chat(d.get("conversation_id")))
    if d.get("execute_at"):
        bits.append(_c("dim", f"at {str(d.get('execute_at'))[11:16]}"))
    if d.get("reason"):
        bits.append(_c("dim", f'"{_preview(d.get("reason"), 40)}"'))
    return " | ".join(bits)


def _r_blocked(d) -> str:
    return (
        _c("red", str(d.get("issues", "")))
        + _c("dim", f' "{_preview(d.get("response_preview"), 40)}"')
    )


def _r_error(d) -> str:
    event = str(d.get("event", "?"))
    error = str(d.get("error", ""))
    return _c("red", f"{event}: {error}")


def _r_generic(d) -> str:
    skip = {"timestamp", "level", "positional_args", "logger_name", "event"}
    bits = []
    for k, v in d.items():
        if k in skip or v is None:
            continue
        rendered = _preview(v, 40)
        bits.append(f"{k}={rendered}")
    return _c("gray", " ".join(bits) or str(d.get("event", "")))


_EVENT_RENDERERS = {
    "processing_message": ("IN", "cyan", _r_in),
    "participation_decision": ("SCORE", "gray", _r_score),
    "social_decision_made": ("DECIDE", "magenta", _r_decide),
    "autonomous_participation": ("DECIDE", "magenta", _r_decide),
    "language_decision": ("LANG", "blue", _r_lang),
    "response_sent": ("OUT", "green", _r_out),
    "reaction_sent": ("OUT", "green", _r_react_out),
    "reaction_seen": ("SEEN", "yellow", _r_reaction_seen),
    "reaction_removed": ("SEEN", "gray", lambda d: f"reaction taken back | msg {d.get('message_id', '?')}"),
    "airi_reacts": ("REACT", "yellow", _r_react_out),
    "memory_retrieval_complete": ("RECALL", "magenta", _r_recall),
    "memory_extracted": ("MEM+", "magenta", _r_mem_new),
    "memories_formed": ("MEM+", "magenta", lambda d: f"{d.get('count', '?')} memories"),
    "memory_superseded": ("MEM~", "magenta", _r_mem_superseded),
    "memory_reinforced_instead_of_duplicate": ("MEM+", "gray", _r_mem_new),
    "memory_reinforced": ("MEM+", "gray", lambda d: f"#{d.get('memory_id', '?')} conf {_num(d.get('confidence'))}"),
    "memory_consolidated": ("MEM+", "magenta", _r_mem_new),
    "memories_formed_nothing": ("MEM", "gray", lambda d: "nothing worth keeping"),
    "memory_formation_nothing_worth_remembering": ("MEM", "gray", lambda d: "nothing worth keeping"),
    "emotional_state_updated": ("STATE", "yellow", _r_state),
    "relationship_updated": ("REL", "yellow", _r_rel),
    "conversation_state_changed": ("LIFE", "blue", _r_life),
    "airi_left_conversation": ("LIFE", "blue", lambda d: _c("red", "left") + _c("dim", f" | {_chat(d.get('conversation_id'))}")),
    "airi_returned": ("LIFE", "blue", lambda d: "returned" + _c("dim", f" | {_chat(d.get('conversation_id'))}")),
    "airi_ignored": ("LIFE", "yellow", lambda d: "no one answered her" + _c("dim", f" | {_chat(d.get('conversation_id'))}")),
    "jealousy_signal": ("JEAL", "magenta", lambda d: f"about {_user(d.get('about_user'))} " + _c("dim", f'"{_preview(d.get("cue"), 30)}"')),
    "topic_set": ("TOPIC", "blue", _r_topic),
    "topic_changed": ("TOPIC", "blue", _r_topic),
    "topic_pending": ("TOPIC", "blue", _r_topic),
    "delayed_action_scheduled": ("TIMER", "yellow", _r_timer),
    "delayed_message_sent": ("OUT", "green", lambda d: "delayed | " + _c("dim", _chat(d.get("conversation_id")))),
    "delayed_reaction_sent": ("OUT", "green", _r_react_out),
    "response_blocked_by_validation": ("BLOCK", "red", _r_blocked),
    "autonomous_response_blocked": ("BLOCK", "red", _r_blocked),
    "reply_dropped_stale": ("DROP", "yellow", lambda d: "conversation moved on -- staying quiet" + _c("dim", f" | {_chat(d.get('chat_id'))}")),
    "llm_skipped_reaction_message": ("SKIP", "gray", lambda d: "reaction message handled locally (no LLM)" + _c("dim", f" | {_chat(d.get('conversation_id'))}")),
    "conversation_summarized": ("MEM+", "magenta", lambda d: f"summary | {d.get('messages', '?')} msgs " + _c("dim", f'"{_preview(d.get("summary_preview"), 44)}"')),
    "follow_up_scheduled": ("TIMER", "yellow", lambda d: "follow-up: " + _c("dim", _preview(d.get("question"), 44))),
    "self_corrected_typo": ("OUT", "green", lambda d: "fixed a typo" + _c("dim", f" | msg {d.get('message_id', '?')}")),
    "llm_structured_retry": ("RETRY", "yellow", lambda d: f"attempt {d.get('attempt', '?')} | {_preview(d.get('error'), 44)}"),
    "llm_empty_content_retry": ("RETRY", "yellow", lambda d: f"empty output (thinking ate the budget) | attempt {d.get('attempt', '?')} | raising max_tokens to {d.get('max_tokens', '?')}"),
    "llm_truncated_retry": ("RETRY", "yellow", lambda d: f"output truncated (budget) | attempt {d.get('attempt', '?')} | raising max_tokens to {d.get('max_tokens', '?')}"),
    "llm_call_done": ("LLM", "gray", lambda d: f"{d.get('seconds', '?')}s | attempt {d.get('attempt', '?')} | out {d.get('completion_tokens', '?')} tok"),
    "llm_skipped_backlog": ("SKIP", "gray", lambda d: f"chat moved on ({d.get('newer_messages', '?')} newer) - no LLM spent"),
    "greeted_back": ("OUT", "green", lambda d: _c("white", f'"{_preview(d.get("text"), 24)}"') + _c("dim", " (local greeting)")),
    "late_typo_fix": ("EDIT", "cyan", lambda d: f"noticed typo {d.get('delay', '?')}s later, edited msg {d.get('message_id', '?')}"),
    "style_audit_complete": ("AUDIT", "yellow", lambda d: f"{d.get('n_messages', '?')} msgs | {len(d.get('alarms', []))} alarms"),
}

# Events whose NAME looks like an error get the error treatment.
_ERROR_MARKERS = ("failed", "error", "_err")


def _pretty_processor(logger, method, event_dict):
    """structlog renderer: one colored, scannable line per event."""
    event = str(event_dict.get("event", ""))

    try:
        time_s = _short_time(event_dict.get("timestamp"))
        renderer = _EVENT_RENDERERS.get(event)
        level = str(event_dict.get("level", "info"))

        is_error = (
            renderer is None
            and (level in ("error", "critical")
                 or any(m in event for m in _ERROR_MARKERS))
        )

        if is_error:
            tag, color = "ERROR", "red"
            body = _r_error(event_dict)
        elif renderer:
            tag, color, fn = renderer
            body = fn(event_dict)
        else:
            # Lifecycle/system-looking events get their own tag so startup
            # and shutdown lines don't read like broken separators.
            if any(m in event for m in ("initializ", "start", "stop", "shut", "bot_", "signal", "database")):
                tag, color = "SYS", "gray"
            else:
                tag, color = "LOG", "gray"
            body = _r_generic(event_dict)

        tag_s = _c(color, _c("bold", f"{tag:<6}"))
        time_s = _c("dim", time_s)
        return f"{time_s} {tag_s} {body}"

    except Exception:
        # Logging must never crash the app.
        try:
            import json
            return json.dumps(event_dict, default=str)
        except Exception:
            return str(event_dict)


_log_listener = None  # keep the queue listener alive globally


def _disable_windows_quickedit():
    """Windows QuickEdit mode freezes ANY console app when text gets
    selected (even by a stray click): all stdout writes block until the
    user presses Enter. For the bot that meant 'stuck, then a burst of
    messages'. Turn it off."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            ENABLE_QUICK_EDIT = 0x0040
            ENABLE_EXTENDED_FLAGS = 0x0080
            new_mode = (mode.value & ~ENABLE_QUICK_EDIT) | ENABLE_EXTENDED_FLAGS
            kernel32.SetConsoleMode(handle, new_mode)
    except Exception:
        pass


def _start_queue_listener(root_logger):
    """Route log writes through a background thread. Even if the console
    blocks for any reason, the event loop keeps running — log lines queue
    and flush instead of freezing the bot."""
    import queue
    from logging.handlers import QueueHandler, QueueListener

    global _log_listener
    log_queue = queue.SimpleQueue()
    queue_handler = QueueHandler(log_queue)

    existing = root_logger.handlers[:]
    root_logger.handlers = [queue_handler]
    _log_listener = QueueListener(
        log_queue, *existing, respect_handler_level=True,
    )
    _log_listener.start()


def configure_logging():
    # Enables ANSI escape sequences on Windows consoles.
    os.system("")

    # Never crash on characters the console can't encode (emoji in message
    # previews, etc.) — replace them instead of raising UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    # Console selection must not freeze the whole bot.
    _disable_windows_quickedit()

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    use_json = getattr(settings, "log_format", "pretty") == "json"

    if use_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = _pretty_processor

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Non-blocking log delivery (see _start_queue_listener).
    _start_queue_listener(logging.getLogger())

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    # B.AI / OpenAI-compatible clients
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str = None):
    return structlog.get_logger(name)
