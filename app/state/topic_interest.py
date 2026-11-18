"""Evolving per-topic interest.

Airi starts with personality-defined interests (Minecraft, Valorant,
brainrot high; boring science, music low). Interests evolve *gradually* from
repeated experience:

    repeated positive/engaging experiences -> interest rises
    repeated boring/negative experiences    -> interest falls

Changes are small and bounded so a single message never swings an interest.
Learned values live in the DB and survive restarts.
"""
from datetime import datetime
from typing import Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.config import settings
from app.database import TopicInterest
from app.state.manager import _TOPIC_STOPWORDS, TOPIC_ANCHORS

logger = structlog.get_logger()


def is_meaningful_topic(topic: str) -> bool:
    """True when the topic is a real subject, not conversational filler.

    Kept consistent with the conversation topic tracker: known anchors are
    always meaningful; single generic/filler words never are. This stops
    discourse junk ('btw', 'airi', 'actually') from polluting Airi's
    evolving interest map.
    """
    key = (topic or "").strip().lower()
    if not key or len(key) < 2:
        return False
    if key in TOPIC_ANCHORS:
        return True
    words = [w for w in key.split() if w not in _TOPIC_STOPWORDS]
    if not words:
        return False
    # A single common/conversational word is not a topic.
    if len(words) == 1 and words[0] in {
        "airi", "hey", "hi", "hello", "yes", "ok", "game", "play", "guy", "guys",
        "man", "girl", "people", "someone", "update", "version", "thing",
    }:
        return False
    return True


class TopicInterestManager:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_topic(self, topic: str) -> Optional[TopicInterest]:
        stmt = select(TopicInterest).where(TopicInterest.topic == topic.strip().lower())
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_interest(self, topic: str) -> float:
        row = await self.get_topic(topic)
        if row is None:
            return 0.5
        return row.interest

    async def get_all(self) -> Dict[str, float]:
        stmt = select(TopicInterest).where(TopicInterest.interest >= 0.4)
        result = await self.session.execute(stmt)
        return {row.topic: row.interest for row in result.scalars().all()}

    async def touch(
        self,
        topic: str,
        engagement: float = 0.0,
        positive: bool = True,
    ) -> float:
        """Gradually adjust interest for a topic after an experience.

        engagement: 0..1 how engaging the moment was.
        positive:   whether the experience was engaging or boring/negative.
        Returns the updated interest value (0.5 when the topic is meaningless).
        """
        key = (topic or "").strip().lower()
        if not is_meaningful_topic(key):
            logger.debug("topic_interest_skipped_filler", candidate=key[:40])
            return 0.5

        row = await self.get_topic(key)
        if row is None:
            row = TopicInterest(topic=key, interest=0.5, exposure_count=0)
            self.session.add(row)

        delta = settings.topic_interest_delta * (0.5 + 0.5 * engagement)
        if not positive:
            delta = -delta

        row.interest = max(settings.topic_interest_min, min(settings.topic_interest_max, row.interest + delta))
        row.exposure_count += 1
        row.last_seen_at = datetime.utcnow()
        row.updated_at = datetime.utcnow()
        await self.session.flush()

        logger.debug(
            "topic_interest_updated",
            topic=key,
            interest=round(row.interest, 3),
            exposure_count=row.exposure_count,
            positive=positive,
        )
        return row.interest

    async def get_interest_map(self) -> Dict[str, float]:
        """Interest map for the attention scorer: all learned interests."""
        return await self.get_all()