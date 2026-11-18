"""Per-conversation output-language statistics, persisted in the DB.

Keeps the 'English is overwhelmingly Airi's language' invariant measurable:
every generated response updates the counters, and the agent consults the
recent Persian usage to keep Persian rare.
"""
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.database import LanguageStats

logger = structlog.get_logger()


class LanguageStatsManager:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create(self, conversation_id: int) -> LanguageStats:
        stmt = select(LanguageStats).where(LanguageStats.conversation_id == conversation_id)
        result = await self.session.execute(stmt)
        stats = result.scalar_one_or_none()
        if stats is None:
            stats = LanguageStats(conversation_id=conversation_id)
            self.session.add(stats)
            await self.session.flush()
        return stats

    async def record_response(self, conversation_id: int, language: str, reason: str, has_snippet: bool) -> LanguageStats:
        stats = await self.get_or_create(conversation_id)
        if language == "persian":
            stats.persian_responses += 1
        else:
            stats.english_responses += 1
        if has_snippet:
            stats.persian_snippets += 1
        stats.last_language = language
        stats.last_persian_reason = reason if language == "persian" else None
        await self.session.flush()

        total = stats.english_responses + stats.persian_responses
        logger.debug(
            "language_stats_updated",
            conversation_id=conversation_id,
            language=language,
            english=stats.english_responses,
            persian=stats.persian_responses,
            english_ratio=round(stats.english_responses / total, 3) if total else 1.0,
        )
        return stats

    async def get_recent_persian_usage(self, conversation_id: int) -> int:
        """Approximate recent Persian usage: how many of the last responses were Persian.

        Derived from the persisted counters' last state — good enough for rate
        limiting Persian frequency without extra tables.
        """
        stats = await self.get_or_create(conversation_id)
        if stats.last_language != "persian":
            return 0
        return min(2, stats.persian_responses)  # persistently damp after any persian response

    async def english_dominance(self, conversation_id: int) -> float:
        """Ratio of English responses (0..1). 1.0 == perfectly English."""
        stats = await self.get_or_create(conversation_id)
        total = stats.english_responses + stats.persian_responses
        if total == 0:
            return 1.0
        return stats.english_responses / total
