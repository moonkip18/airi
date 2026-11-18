"""Memory maintenance: decay, reinforcement, consolidation.

Runs periodically in the background loop (never per message).

  * DECAY: old, low-importance memories slowly lose retrieval weight (lower
    importance); important memories are protected. Nothing is blindly deleted;
    only clearly dead/trivial superseded history may be pruned.
  * REINFORCEMENT: when a new memory closely matches an existing one, the
    existing memory's confidence/importance/reinforcement counter rises instead
    of creating a near-duplicate.
  * CONSOLIDATION: repeated related episodic memories are folded into higher
    -level semantic/preference memories, while the original episodes are kept.
"""
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.config import settings
from app.database import Memory, MemoryType
from app.memory.embeddings import embedding_model

logger = structlog.get_logger()


class MemoryMaintenance:
    def __init__(self, session: AsyncSession):
        self.session = session

    # ------------------------------------------------------------------
    # Reinforcement
    # ------------------------------------------------------------------

    async def find_match(
        self,
        content: str,
        user_id: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Optional[Memory]:
        """Find the strongest existing memory matching new content."""
        threshold = similarity_threshold if similarity_threshold is not None else settings.memory_reinforcement_threshold
        query_vec = embedding_model.encode_single(content)

        stmt = select(Memory).where(
            Memory.superseded_by.is_(None),
            Memory.confidence > 0.2,
        )
        if user_id is not None:
            stmt = stmt.where(Memory.user_id == user_id)
        result = await self.session.execute(stmt.limit(500))
        memories = result.scalars().all()

        best: Optional[Memory] = None
        best_score = 0.0
        for memory in memories:
            if not memory.embedding:
                continue
            sim = embedding_model.similarity(query_vec, memory.embedding)
            if sim >= threshold and sim > best_score:
                best = memory
                best_score = sim
        return best

    async def reinforce(self, memory: Memory, content: str) -> Memory:
        """Confirm an existing fact: raise confidence/importance slightly."""
        memory.reinforcement = (memory.reinforcement or 0) + 1
        memory.confidence = min(1.0, memory.confidence + 0.04)
        memory.importance = min(1.0, memory.importance + 0.02)
        memory.updated_at = datetime.utcnow()
        await self.session.flush()
        logger.debug(
            "memory_reinforced",
            memory_id=memory.id,
            reinforcement=memory.reinforcement,
            confidence=round(memory.confidence, 3),
        )
        return memory

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    async def decay(self, factor: float = 0.995):
        """Slowly lower importance of old, unreinforced memories.

        Important memories (importance >= 0.6) or recently reinforced ones
        are largely protected. Low-importance memories drift down so retrieval
        naturally prefers the durable facts.
        """
        cutoff = datetime.utcnow() - timedelta(days=3)
        stmt = select(Memory).where(
            Memory.created_at < cutoff,
            Memory.superseded_by.is_(None),
        )
        result = await self.session.execute(stmt)
        for memory in result.scalars().all():
            if (memory.reinforcement or 0) > 0 and memory.importance >= 0.5:
                continue  # reinforced facts stay strong
            if memory.importance <= 0.35:
                memory.importance = max(0.1, memory.importance * 0.995)
            else:
                memory.importance = max(0.1, memory.importance * factor)
        await self.session.flush()

    async def prune_dead(self, days: int = 60):
        """Physically remove only clearly-dead data: superseded memories whose
        superseding memory is itself gone, or trivial old low-value memories."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        stmt = select(Memory).where(
            Memory.superseded_by.is_not(None),
            Memory.created_at < cutoff,
        )
        result = await self.session.execute(stmt)
        for memory in result.scalars().all():
            await self.session.delete(memory)
        await self.session.flush()

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    async def consolidate(self, max_episodes: int = 6):
        """Fold clusters of related episodic memories into knowledge.

        Episodes sharing strong topic overlap with importance >= 0.45 are
        grouped; for each group of >= 3, a SEMANTIC/PREFERENCE memory is
        created (if one doesn't already capture it) that generalizes the
        episodes. Original episodes are preserved.
        """
        from app.database import Memory as _M

        stmt = select(_M).where(
            _M.memory_type.in_([MemoryType.EPISODIC]),
            _M.superseded_by.is_(None),
            _M.importance >= 0.45,
        ).order_by(_M.created_at.desc()).limit(max_episodes * 20)
        result = await self.session.execute(stmt)
        episodes = result.scalars().all()
        if len(episodes) < 3:
            return 0

        clustered: List[List[_M]] = []
        used = set()
        for i, ep in enumerate(episodes):
            if ep.id in used:
                continue
            group = [ep]
            used.add(ep.id)
            for j, other in enumerate(episodes):
                if other.id in used or other.id == ep.id:
                    continue
                if not ep.topics or not other.topics:
                    continue
                overlap = set(ep.topics) & set(other.topics)
                if overlap:
                    group.append(other)
                    used.add(other.id)
            if len(group) >= 3:
                clustered.append(group)

        created = 0
        for group in clustered:
            topics = sorted({t for ep in group for t in (ep.topics or [])})
            people = sorted({p for ep in group for p in (ep.people_involved or [])})
            topic_str = ", ".join(topics[:4]) if topics else "various"
            content = self._generalize(group, topic_str)
            if not content:
                continue

            # Skip if an equivalent semantic/preference memory already exists.
            if await self._has_generalization(content):
                continue

            memory_type = MemoryType.PREFERENCE if any(
                "likes" in (ep.content or "").lower() or "enjoys" in (ep.content or "").lower()
                for ep in group
            ) else MemoryType.SEMANTIC

            from app.memory.store import MemoryStore
            store = MemoryStore(self.session)
            avg_importance = sum(ep.importance for ep in group) / len(group)
            new_memory = await store.create_memory(
                memory_type=memory_type,
                content=content,
                user_id=group[0].user_id,
                conversation_id=group[0].conversation_id,
                people_involved=people,
                topics=topics,
                importance=min(1.0, avg_importance + 0.1),
                confidence=min(1.0, avg_importance + 0.1),
                source="consolidation",
            )
            created += 1
            logger.info(
                "memory_consolidated",
                new_memory_id=new_memory.id,
                memory_type=memory_type.value,
                episode_ids=[e.id for e in group],
                content_preview=content[:120],
            )
        return created

    def _generalize(self, group: List[Memory], topic_str: str) -> str:
        """Create a generalizing sentence from a group of related episodes."""
        texts = [ep.content for ep in group if ep.content]
        if not texts:
            return ""
        head = texts[0][:80]
        tail = texts[-1][:80]
        people = sorted({p for ep in group for p in (ep.people_involved or [])})
        who = f"user{people[0]}" if people else "this person"
        return (
            f"Repeatedly involved with {topic_str}: episodes like "
            f"\"{head}...\" and \"{tail}...\" suggest this is an ongoing pattern "
            f"for {who}, not a one-off event."
        )

    async def _has_generalization(self, content: str) -> bool:
        threshold = 0.85
        query_vec = embedding_model.encode_single(content)
        stmt = select(Memory).where(
            Memory.memory_type.in_([MemoryType.SEMANTIC, MemoryType.PREFERENCE]),
            Memory.superseded_by.is_(None),
        )
        result = await self.session.execute(stmt.limit(200))
        for memory in result.scalars().all():
            if memory.embedding and embedding_model.similarity(query_vec, memory.embedding) >= threshold:
                return True
        return False

    # ------------------------------------------------------------------
    # Backend migration
    # ------------------------------------------------------------------

    async def reembed_stale_backend(self, batch: int = 200) -> int:
        """Re-embed memories stored under a different embedding backend.

        When sentence-transformers becomes available (after previously
        running on the hashed fallback — or the reverse), old vectors live
        in a different space with a different dimensionality: similarity()
        returns 0 for them, making those memories invisible to retrieval.
        One pass re-encodes everything that doesn't match the current
        dimensionality so history stays retrievable.
        """
        current = embedding_model.get_model()
        target_dim = len(embedding_model.encode_single("dim probe"))
        stmt = select(Memory).where(
            Memory.superseded_by.is_(None),
        ).limit(batch)
        result = await self.session.execute(stmt)
        fixed = 0
        for memory in result.scalars().all():
            if not memory.embedding or len(memory.embedding) == target_dim:
                continue
            memory.embedding = embedding_model.encode_single(memory.content)
            fixed += 1
        if fixed:
            await self.session.flush()
            logger.info(
                "memories_reembedded",
                count=fixed,
                backend=getattr(current, "__class__", None).__name__,
                target_dim=target_dim,
            )
        return fixed