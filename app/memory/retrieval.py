from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from sqlalchemy import select, and_, or_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from app.database import (
    Memory, MemoryType, Conversation, ConversationStateModel,
    ConversationState, User, Relationship
)
from app.memory.embeddings import embedding_model
from app.memory.store import MemoryStore
from app.config import settings

logger = structlog.get_logger()


@dataclass
class RetrievalContext:
    query_text: str
    conversation_id: Optional[int] = None
    user_id: Optional[int] = None
    current_topic: Optional[str] = None
    recent_participants: Optional[List[int]] = None
    emotional_state: Optional[Dict[str, float]] = None
    relationship_context: Optional[Dict[int, Dict[str, float]]] = None
    max_results: int = 5
    # People/entities relevant to the current moment: the speaker PLUS anyone
    # referenced/mentioned/active. The speaker is not the sole memory namespace.
    relevant_user_ids: Optional[List[int]] = None


@dataclass
class RetrievedMemory:
    memory: Memory
    relevance_score: float
    semantic_similarity: float
    importance_score: float
    recency_score: float
    relationship_relevance: float
    redundancy_penalty: float


class MemoryRetriever:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.store = MemoryStore(session)

    async def retrieve(self, context: RetrievalContext) -> List[RetrievedMemory]:
        base_memories = await self._candidate_retrieval(context)
        if not base_memories:
            return []

        scored = await self._score_and_rerank(base_memories, context)
        scored.sort(key=lambda x: x.relevance_score, reverse=True)

        final = self._deduplicate(scored, context.max_results)

        logger.info(
            "memory_retrieval_complete",
            query_preview=context.query_text[:100],
            candidates=len(base_memories),
            returned=len(final),
            top_score=final[0].relevance_score if final else 0,
        )

        return final

    async def _candidate_retrieval(self, context: RetrievalContext) -> List[Memory]:
        query_parts = [context.query_text]

        if context.current_topic:
            query_parts.append(context.current_topic)

        if context.recent_participants:
            participant_topics = await self._get_participant_topics(context.recent_participants)
            query_parts.extend(participant_topics)

        combined_query = " ".join(query_parts)

        if not await self._has_memories(context):
            return []

        # Retrieve across every relevant person (plus general knowledge), so the
        # speaker is not the sole namespace.
        relevant = context.relevant_user_ids or ([context.user_id] if context.user_id else None)

        result = await self.store.retrieve_memories(
            query_text=combined_query,
            conversation_id=context.conversation_id,
            user_id=context.user_id,
            user_ids=relevant,
            limit=context.max_results * 4,
            min_importance=0.1,
            min_confidence=0.3,
            exclude_superseded=True,
        )
        return [m for m, _ in result]

    async def _has_memories(self, context: RetrievalContext) -> bool:
        stmt = select(func.count(Memory.id)).where(
            and_(
                Memory.superseded_by.is_(None),
                Memory.confidence >= 0.3,
            )
        )
        if context.relevant_user_ids:
            relevant = [u for u in context.relevant_user_ids if u is not None]
            if relevant:
                stmt = stmt.where(or_(Memory.user_id.in_(relevant), Memory.user_id.is_(None)))
        elif context.user_id:
            stmt = stmt.where(or_(Memory.user_id == context.user_id, Memory.user_id.is_(None)))

        result = await self.session.execute(stmt)
        return result.scalar() > 0

    async def _get_participant_topics(self, participant_ids: List[int]) -> List[str]:
        """Topics previously associated with the active participants.

        people_involved is a JSON column, so the PostgreSQL-array .overlap()
        operator is invalid here — filter in Python instead (DB-agnostic).
        """
        stmt = select(Memory).where(
            and_(
                Memory.memory_type.in_([MemoryType.SEMANTIC, MemoryType.PREFERENCE]),
                Memory.superseded_by.is_(None),
            )
        ).limit(200)
        result = await self.session.execute(stmt)
        ps = set(participant_ids)
        all_topics = []
        for memory in result.scalars().all():
            if ps & set(memory.people_involved or []):
                all_topics.extend(memory.topics or [])
        return list(set(all_topics))[:10]

    async def _score_and_rerank(
        self,
        memories: List[Memory],
        context: RetrievalContext,
    ) -> List[RetrievedMemory]:
        query_embedding = embedding_model.encode_single(context.query_text)

        scored = []
        seen_content = set()

        for memory in memories:
            if not memory.embedding:
                continue

            semantic_sim = embedding_model.similarity(query_embedding, memory.embedding)

            content_key = memory.content[:100].lower()
            if content_key in seen_content:
                redundancy = 0.5
            else:
                redundancy = 0.0
                seen_content.add(content_key)

            importance_score = memory.importance

            recency_score = self._compute_recency(memory.created_at)

            relationship_relevance = self._compute_relationship_relevance(
                memory, context.recent_participants, context.relationship_context
            )

            relevance = (
                semantic_sim * 0.45
                + importance_score * 0.2
                + recency_score * 0.15
                + relationship_relevance * 0.15
                - redundancy * 0.1
            )

            # Memories formed in the CURRENT conversation rank slightly above
            # equally-relevant ones from other chats — but everything is still
            # in the candidate pool, so long-term knowledge travels with her.
            if context.conversation_id is not None and memory.conversation_id == context.conversation_id:
                relevance += 0.05

            # People weighting: the speaker keeps a modest priority, and anyone
            # referenced/mentioned in the moment also ranks up — but neither is
            # absolute, so Sarah can still surface Alex's memory.
            relevant = set(context.relevant_user_ids or [])
            if context.user_id is not None and memory.user_id == context.user_id:
                relevance += 0.05
            elif memory.user_id in relevant:
                relevance += 0.03
            elif memory.people_involved and relevant & set(memory.people_involved):
                relevance += 0.03

            scored.append(RetrievedMemory(
                memory=memory,
                relevance_score=relevance,
                semantic_similarity=semantic_sim,
                importance_score=importance_score,
                recency_score=recency_score,
                relationship_relevance=relationship_relevance,
                redundancy_penalty=redundancy,
            ))

        return scored

    def _compute_recency(self, created_at: datetime) -> float:
        age_hours = (datetime.utcnow() - created_at).total_seconds() / 3600
        if age_hours < 1:
            return 1.0
        elif age_hours < 6:
            return 0.9
        elif age_hours < 24:
            return 0.7
        elif age_hours < 168:
            return 0.5
        elif age_hours < 720:
            return 0.3
        else:
            return 0.15

    def _compute_relationship_relevance(
        self,
        memory: Memory,
        recent_participants: Optional[List[int]],
        relationship_context: Optional[Dict[int, Dict[str, float]]],
    ) -> float:
        if not recent_participants or not memory.people_involved:
            return 0.0

        overlap = set(recent_participants) & set(memory.people_involved)
        if not overlap:
            return 0.0

        base_score = len(overlap) * 0.2

        if relationship_context:
            for pid in overlap:
                rel = relationship_context.get(pid, {})
                if rel.get("affection", 0) > 0.5:
                    base_score += 0.1
                if rel.get("trust", 0) > 0.5:
                    base_score += 0.05

        return min(base_score, 1.0)

    def _deduplicate(
        self,
        scored: List[RetrievedMemory],
        max_results: int,
    ) -> List[RetrievedMemory]:
        if not scored:
            return []

        final = [scored[0]]
        seen_topics = set(scored[0].memory.topics)

        for item in scored[1:]:
            if len(final) >= max_results:
                break

            memory_topics = set(item.memory.topics)
            topic_overlap = len(seen_topics & memory_topics) / max(len(memory_topics), 1)

            if topic_overlap > 0.7 and item.relevance_score < final[-1].relevance_score * 0.8:
                continue

            final.append(item)
            seen_topics.update(memory_topics)

        return final


async def _resolve_referenced_people(
    session: AsyncSession,
    message_text: str,
    candidate_user_ids: List[int],
) -> List[int]:
    """Find which known users are *named* in the current message.

    Cheap entity resolution: match conversation participants' usernames and
    first names as word tokens in the message. Lets "remember when Alex passed
    that exam?" point retrieval at Alex even though Sarah is speaking.
    """
    import re as _re
    text_lower = (message_text or "").lower()
    if not text_lower or not candidate_user_ids:
        return []

    referenced = []
    for uid in set(candidate_user_ids):
        if uid is None:
            continue
        stmt = select(User).where(User.id == uid)
        user = (await session.execute(stmt)).scalar_one_or_none()
        if user is None:
            continue
        for name in (user.username, user.first_name):
            if not name or len(name) < 3:
                continue
            if _re.search(rf"(?<!\w){_re.escape(name.lower())}(?!\w)", text_lower):
                referenced.append(uid)
                break
    return referenced


async def build_retrieval_context(
    session: AsyncSession,
    conversation_id: int,
    user_id: int,
    message_text: str,
    recent_messages: List[Dict[str, Any]],
    mentioned_user_ids: Optional[List[int]] = None,
) -> RetrievalContext:
    from app.memory.store import ensure_emotional_state
    from app.database import Conversation
    from sqlalchemy.orm import selectinload

    # Load the actual conversation by primary key (conversation_id here is the
    # internal DB id, not the Telegram chat id), eagerly loading its state.
    conversation = (
        await session.execute(
            select(Conversation).options(selectinload(Conversation.state)).where(Conversation.id == conversation_id)
        )
    ).scalar_one_or_none()
    state = conversation.state if conversation else None

    emotional_state_obj = await ensure_emotional_state(session)
    emotional_state = {
        "mood": emotional_state_obj.mood,
        "energy": emotional_state_obj.energy,
        "curiosity": emotional_state_obj.curiosity,
        "amusement": emotional_state_obj.amusement,
        "annoyance": emotional_state_obj.annoyance,
        "interest": emotional_state_obj.interest,
        "social_engagement": emotional_state_obj.social_engagement,
    }

    recent_participants = [msg["user_id"] for msg in recent_messages[-10:] if msg.get("user_id")]

    # Relevant people = speaker + explicitly mentioned + anyone NAMED among the
    # conversation's members + recent active participants. Name resolution runs
    # against ALL conversation participants (a third party can be recalled even
    # if they never spoke in the current window). This is the recall namespace.
    from app.database import ConversationParticipant
    participant_rows = (
        await session.execute(
            select(ConversationParticipant.user_id).where(ConversationParticipant.conversation_id == conversation_id)
        )
    ).scalars().all()
    candidates = set(participant_rows) | set(recent_participants) | set(mentioned_user_ids or [])
    referenced = await _resolve_referenced_people(session, message_text, candidates)
    relevant_user_ids = list({user_id} | set(mentioned_user_ids or []) | set(referenced) | set(recent_participants))

    relationship_context = {}
    for pid in set(recent_participants) | set(relevant_user_ids):
        stmt = select(Relationship).where(Relationship.user_id == pid)
        result = await session.execute(stmt)
        rel = result.scalar_one_or_none()
        if rel:
            relationship_context[pid] = {
                "affection": rel.affection,
                "trust": rel.trust,
                "respect": rel.respect,
                "annoyance": rel.annoyance,
                "familiarity": rel.familiarity,
                "interest": rel.interest,
            }

    return RetrievalContext(
        query_text=message_text,
        conversation_id=conversation_id,
        user_id=user_id,
        current_topic=state.current_topic if state else None,
        recent_participants=recent_participants,
        emotional_state=emotional_state,
        relationship_context=relationship_context,
        max_results=settings.max_memory_results,
        relevant_user_ids=relevant_user_ids,
    )