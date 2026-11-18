"""Shared small helpers.

_age_label: human-scale age of a timestamp. Memory without a WHEN reads
like a database row; with it, it reads like remembering. Used by both the
memory pipeline (core.py) and the prompt renderer (context.py) — kept here
to avoid a circular import between them.
"""
from datetime import datetime
from typing import Optional


def _age_label(dt: Optional[datetime]) -> str:
    if dt is None:
        return "a while ago"
    delta = (datetime.utcnow() - dt).total_seconds()
    if delta < 3600:
        return "just now"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    days = delta / 86400.0
    if days < 7:
        return f"{int(days)}d ago"
    if days < 30:
        return f"{int(days // 7)}w ago"
    months = days / 30.0
    if months < 12:
        return f"{int(months)}mo ago"
    return f"{int(months / 12)}y ago"
