from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import select, and_, or_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from app.database import (
    Memory, MemoryType, User, Conversation, Relationship,
    ConversationStateModel, ConversationState, EmotionalState,
    get_session
)
from app.memory.embeddings import embedding_model
from app.config import settings

logger = structlog.get_logger()


class MemoryStore:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_memory(
        self,
        memory_type: MemoryType,
        content: str,
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None,
        people_involved: Optional[List[int]] = None,
        topics: Optional[List[str]] = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        source: str = "observation",
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        superseded_by: Optional[int] = None,
        reinforcement: int = 0,
    ) -> Memory:
        embedding_text = f"{content} {' '.join(topics or [])}"
        embedding = embedding_model.encode_single(embedding_text)

        memory = Memory(
            memory_type=memory_type,
            content=content,
            user_id=user_id,
            conversation_id=conversation_id,
            people_involved=people_involved or [],
            topics=topics or [],
            importance=importance,
            confidence=confidence,
            source=source,
            embedding=embedding,
            valid_from=valid_from,
            valid_until=valid_until,
            superseded_by=superseded_by,
            reinforcement=reinforcement,
        )
        self.session.add(memory)
        await self.session.flush()
        return memory

    async def get_memory(self, memory_id: int) -> Optional[Memory]:
        result = await self.session.execute(select(Memory).where(Memory.id == memory_id))
        return result.scalar_one_or_none()

    async def get_memory_by_uuid(self, memory_uuid: str) -> Optional[Memory]:
        result = await self.session.execute(select(Memory).where(Memory.memory_id == memory_uuid))
        return result.scalar_one_or_none()

    async def update_memory(self, memory: Memory, **kwargs) -> Memory:
        for key, value in kwargs.items():
            if hasattr(memory, key):
                setattr(memory, key, value)
        memory.updated_at = datetime.utcnow()
        await self.session.flush()
        return memory

    async def supersede_memory(self, old_memory: Memory, new_memory: Memory) -> Memory:
        old_memory.superseded_by = new_memory.id
        old_memory.confidence = max(0.0, old_memory.confidence - 0.3)
        old_memory.updated_at = datetime.utcnow()
        await self.session.flush()
        return old_memory

    async def retrieve_memories(
        self,
        query_text: str,
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None,
        memory_types: Optional[List[MemoryType]] = None,
        people_involved: Optional[List[int]] = None,
        topics: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        min_confidence: float = 0.0,
        exclude_superseded: bool = True,
        user_ids: Optional[List[int]] = None,
    ) -> List[Tuple[Memory, float]]:
        query_embedding = embedding_model.encode_single(query_text)

        conditions = [
            Memory.importance >= min_importance,
            Memory.confidence >= min_confidence,
        ]

        if exclude_superseded:
            conditions.append(Memory.superseded_by.is_(None))

        # People/entity scoping. The speaker must NOT be the sole namespace: when
        # relevant people are supplied we retrieve across all of them (plus any
        # general, non-user-scoped semantic knowledge). `user_ids` lets callers
        # pass multiple relevant people; plain `user_id` stays for backwards compat.
        if user_ids is not None:
            relevant = list(set(u for u in user_ids if u is not None))
            if relevant:
                conditions.append(
                    or_(Memory.user_id.in_(relevant), Memory.user_id.is_(None))
                )
        elif user_id is not None:
            conditions.append(or_(Memory.user_id == user_id, Memory.user_id.is_(None)))

        # NOTE: no conversation_id filter here. Long-term memory must be
        # recallable across conversations — what she learned about someone in
        # chat A is exactly what makes her feel persistent in chat B. The
        # current conversation gets a small ranking bonus in the retriever.

        if memory_types:
            conditions.append(Memory.memory_type.in_(memory_types))

        stmt = select(Memory).where(and_(*conditions)).order_by(desc(Memory.importance), desc(Memory.created_at))
        result = await self.session.execute(stmt.limit(limit * 6))
        memories = result.scalars().all()

        # people_involved/topics are JSON columns (SQLite) — overlap() is a
        # PostgreSQL-ARRAY op and is invalid here, so we filter in Python.
        if people_involved:
            pi = set(people_involved)
            memories = [m for m in memories if pi & set(m.people_involved or [])]
        if topics:
            tp = {t.lower() for t in topics}
            memories = [m for m in memories if tp & {t.lower() for t in (m.topics or [])}]

        if not memories:
            return []

        scored_memories = []
        for memory in memories:
            if memory.embedding:
                similarity = embedding_model.similarity(query_embedding, memory.embedding)
            else:
                similarity = 0.0

            recency_score = self._recency_score(memory.created_at)
            importance_score = memory.importance
            confidence_score = memory.confidence

            final_score = (
                similarity * 0.5
                + importance_score * 0.2
                + confidence_score * 0.15
                + recency_score * 0.15
            )

            if people_involved and memory.people_involved:
                overlap = len(set(people_involved) & set(memory.people_involved))
                final_score += overlap * 0.1

            scored_memories.append((memory, final_score))

        scored_memories.sort(key=lambda x: x[1], reverse=True)

        for memory, _ in scored_memories[:limit]:
            memory.last_retrieved = datetime.utcnow()
            memory.retrieval_count += 1

        await self.session.flush()
        return scored_memories[:limit]

    def _recency_score(self, created_at: datetime) -> float:
        age_hours = (datetime.utcnow() - created_at).total_seconds() / 3600
        if age_hours < 1:
            return 1.0
        elif age_hours < 24:
            return 0.8
        elif age_hours < 168:
            return 0.5
        elif age_hours < 720:
            return 0.3
        else:
            return 0.1

    async def get_recent_memories(
        self,
        conversation_id: int,
        hours: int = 24,
        limit: int = 20,
    ) -> List[Memory]:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        stmt = select(Memory).where(
            and_(
                Memory.conversation_id == conversation_id,
                Memory.created_at >= cutoff,
                Memory.superseded_by.is_(None),
            )
        ).order_by(desc(Memory.created_at)).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def find_contradictory_memories(
        self,
        content: str,
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None,
        similarity_threshold: float = 0.8,
    ) -> List[Memory]:
        """Find memories the new content contradicts.

        Deliberately searches across ALL conversations for the person: a fact
        learned in chat A must still be correctable by new information in
        chat B, or she'd hold two contradictory versions of the same fact.
        """
        query_embedding = embedding_model.encode_single(content)

        conditions = [
            Memory.superseded_by.is_(None),
            Memory.confidence > 0.5,
        ]

        if user_id:
            conditions.append(Memory.user_id == user_id)

        stmt = select(Memory).where(and_(*conditions))
        result = await self.session.execute(stmt)
        memories = result.scalars().all()

        contradictory = []
        for memory in memories:
            if memory.embedding:
                sim = embedding_model.similarity(query_embedding, memory.embedding)
                if sim >= similarity_threshold:
                    contradictory.append(memory)

        return contradictory


class MemoryFormation:
    """Forms long-term memories from conversation windows.

    Uses ONE batched LLM call per conversation window (not one per message)
    to decide what — if anything — is worth remembering. Explicitly supports
    'DO NOT REMEMBER'. New memories that contradict existing ones supersede
    them (temporal validity + confidence reduction), preserving a correction
    trail rather than silently overwriting history.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.store = MemoryStore(session)

    async def extract_from_conversation(
        self,
        conversation_id: int,
        messages: List[Dict[str, Any]],
        usernames_by_id: Optional[Dict[int, str]] = None,
        persona_name: Optional[str] = None,
    ) -> List[Memory]:
        if not messages:
            return []

        transcript = self._build_transcript(messages)
        if not transcript.strip():
            return []

        usernames_by_id = usernames_by_id or {}
        outcome = await self._analyze_window(
            transcript, usernames_by_id, persona_name,
        )
        if outcome is None or not outcome.should_remember:
            logger.info(
                "memory_formation_nothing_worth_remembering",
                conversation_id=conversation_id,
                messages_analyzed=len(messages),
            )
            return []

        # Persistence phase (fast local reads/writes + embedding computation)
        # runs under the global write lock, AND COMMITS INSIDE IT. The commit
        # must never be deferred past the lock release: an open uncommitted
        # write transaction holds SQLite's RESERVED lock, so every other
        # writer then blocks the full 30s busy_timeout and fails with
        # "database is locked" — the 14:02:45→14:04:24 cascade. The LLM call
        # above stayed OUTSIDE the lock (it must never hold it).
        from app.database.session import db_write_lock
        async with db_write_lock:
            created = await self._persist_extraction(
                conversation_id, outcome, usernames_by_id,
            )
            await self.session.commit()
        return created

    async def _persist_extraction(
        self,
        conversation_id: int,
        outcome,
        usernames_by_id: Dict[int, str],
    ) -> List[Memory]:
        created: List[Memory] = []
        for item in outcome.memories[: settings.max_memories_per_extraction]:
            if item.importance < settings.memory_importance_threshold:
                continue

            memory_type_map = {
                "episodic": MemoryType.EPISODIC,
                "semantic": MemoryType.SEMANTIC,
                "relationship": MemoryType.RELATIONSHIP,
                "preference": MemoryType.PREFERENCE,
                "behavioral_pattern": MemoryType.BEHAVIORAL_PATTERN,
            }
            memory_type = memory_type_map.get(item.memory_type)
            if memory_type is None:
                continue

            # Reinforcement: if this "new" fact closely matches an existing
            # durable memory, strengthen it instead of storing a duplicate.
            if memory_type in (MemoryType.SEMANTIC, MemoryType.PREFERENCE, MemoryType.RELATIONSHIP):
                from app.memory.maintenance import MemoryMaintenance
                maintenance = MemoryMaintenance(self.session)
                match = await maintenance.find_match(
                    item.content, user_id=item.user_id,
                )
                if match is not None:
                    await maintenance.reinforce(match, item.content)
                    logger.info(
                        "memory_reinforced_instead_of_duplicate",
                        existing_memory_id=match.id,
                        new_content_preview=item.content[:100],
                    )
                    continue

            memory = await self.store.create_memory(
                memory_type=memory_type,
                content=item.content,
                user_id=item.user_id,
                conversation_id=conversation_id,
                people_involved=item.people_involved,
                topics=item.topics,
                importance=item.importance,
                confidence=item.confidence,
                source="extraction",
            )

            # Contradiction handling: an old memory that conflicts with this
            # new one gets superseded (corrected), never silently replaced.
            contradictory = await self.store.find_contradictory_memories(
                content=item.content,
                user_id=item.user_id,
                conversation_id=conversation_id,
            )
            for old in contradictory:
                if old.id == memory.id:
                    continue
                await self.store.supersede_memory(old, memory)
                logger.info(
                    "memory_superseded",
                    old_memory_id=old.id,
                    new_memory_id=memory.id,
                    old_content=old.content[:100],
                    new_content=item.content[:100],
                )

            created.append(memory)
            logger.info(
                "memory_extracted",
                memory_id=memory.id,
                type=item.memory_type,
                importance=item.importance,
                content_preview=item.content[:100],
            )

            # Humans remember to ask about things later ("how did the exam
            # go?"). Important memories with a future dimension get ONE
            # scheduled follow-up question, hours ahead.
            if (
                item.importance >= 0.75
                and (item.follow_up_question or "").strip()
            ):
                await self._schedule_follow_up(
                    conversation_id, memory, item.follow_up_question.strip(),
                )

        # Commitments: her own word is a memory with a deadline. Stored like
        # any other memory; if the extractor offered a natural check-up
        # question, it gets scheduled hours ahead.
        for item in outcome.commitments[: settings.max_memories_per_extraction]:
            content = (item.content or "").strip()
            if not content:
                continue
            memory = await self.store.create_memory(
                memory_type=MemoryType.EPISODIC,
                content=content,
                user_id=item.user_id,
                conversation_id=conversation_id,
                people_involved=item.people_involved,
                topics=item.topics,
                importance=max(0.65, item.importance),
                confidence=0.9,
                source="commitment",
            )
            created.append(memory)
            logger.info(
                "commitment_recorded",
                memory_id=memory.id,
                content_preview=content[:80],
            )
            if (item.follow_up_question or "").strip():
                await self._schedule_follow_up(
                    conversation_id, memory, item.follow_up_question.strip(),
                )

        # Stances: her stated opinions persist in the self-model so her
        # takes stay consistent across days ("people notice flippers").
        if outcome.stances:
            from app.agent.learning import SelfModelManager

            model = await SelfModelManager(self.session).get_or_create()
            stances = dict(model.stances or {})
            for s in outcome.stances[:3]:
                topic = (s.topic or "").strip().lower()[:60]
                stance = (s.stance or "").strip()[:200]
                if topic and stance:
                    stances[topic] = stance
            if len(stances) > 10:
                stances = dict(list(stances.items())[-10:])
            model.stances = stances
            model.updated_at = datetime.utcnow()
            await self.session.flush()

        return created

    async def _schedule_follow_up(
        self,
        conversation_id: int,
        memory,
        question: str,
    ):
        from app.social.delayed import DelayedActionScheduler
        import random as _random

        scheduler = DelayedActionScheduler(self.session)
        # Don't stack up questions she never asked.
        if await scheduler.count_pending(conversation_id) >= 2:
            return

        await scheduler.schedule_in(
            conversation_id,
            "follow_up",
            delay_seconds=_random.uniform(3 * 3600, 18 * 3600),
            payload={"text": question[:300], "memory_id": memory.memory_id},
            reason=f"remembered to ask: {question[:80]}",
        )
        logger.info(
            "follow_up_scheduled",
            conversation_id=conversation_id,
            memory_id=memory.id,
            question=question[:60],
        )

    def _build_transcript(self, messages: List[Dict[str, Any]]) -> str:
        lines = []
        for msg in messages:
            username = msg.get("username") or f"user_{msg.get('user_id', '?')}"
            text = (msg.get("text") or "").strip()
            if text:
                lines.append(f"{username}: {text}")
        return "\n".join(lines)

    async def _analyze_window(
        self,
        transcript: str,
        usernames_by_id: Dict[int, str],
        persona_name: Optional[str] = None,
    ):
        """Single batched LLM call per conversation window."""
        import json
        from typing import Literal as _Literal
        from pydantic import BaseModel as _BaseModel, Field as _Field
        from app.llm import llm_provider

        class ExtractedMemoryItem(_BaseModel):
            memory_type: _Literal["episodic", "semantic", "relationship", "preference", "behavioral_pattern"] = "episodic"
            content: str = ""
            # The ONE person this fact is primarily about (who did/said/felt it).
            about_username: str = ""
            about_usernames: List[str] = _Field(default_factory=list)
            people_involved: List[int] = _Field(default_factory=list)
            topics: List[str] = _Field(default_factory=list)
            importance: float = 0.0
            confidence: float = 0.5
            # Optional natural follow-up question for later ("how did the exam
            # go?"). Empty = no follow-up.
            follow_up_question: str = ""
            user_id: Optional[int] = None  # resolved locally from usernames

        class StanceItem(_BaseModel):
            topic: str = ""
            stance: str = ""

        class ExtractionOutcome(_BaseModel):
            should_remember: bool = False
            memories: List[ExtractedMemoryItem] = _Field(default_factory=list)
            # Her own commitments/promises made IN this window ("I'll send it
            # tomorrow") — captured so she can follow through later.
            commitments: List[ExtractedMemoryItem] = _Field(default_factory=list)
            # Stances she took (topic -> short opinion). Only STRONG takes.
            stances: List[StanceItem] = _Field(default_factory=list)

        system = (
            "You extract long-term memories from a group-chat transcript for an "
            "AI persona. Respond ONLY with valid JSON matching the schema. "
            "Most chat windows contain NOTHING worth remembering long-term — "
            "prefer should_remember=false unless there is genuinely durable "
            "information (stable preferences, facts about people, notable shared "
            "events, relationship dynamics, recurring behavioral patterns). "
            "Never remember transient jokes, fleeting emotions, or small talk. "
            "The persona's OWN messages appear in the transcript under her name: "
            "extract facts about her the same way (her commitments/promises go "
            "to `commitments`, her strong opinions to `stances`, everything else "
            "regular). ALWAYS attribute facts to the exact person they are "
            "about — two different usernames are two different people, never "
            "merge their histories. Write memory content in English."
        )

        prompt = (
            f"Transcript:\n{transcript}\n\n"
            f"username -> user_id map: "
            f"{json.dumps({str(k): v for k, v in usernames_by_id.items()})}\n\n"
            + (
                f"The persona herself is \"{persona_name}\" — messages starting "
                f"with \"{persona_name}:\" are HERS. Her commitments/promises go "
                "to `commitments`, her strong opinions to `stances`.\n\n"
                if persona_name else ""
            ) +
            "For each memory: about_username = the ONE person the fact is "
            "primarily about (the person who did, said, or felt it — usually "
            "the speaker revealing something about themselves). about_usernames "
            "= everyone involved. follow_up_question = only for important "
            "memories with a future dimension (upcoming events, ongoing "
            "situations): a short casual question to ask that person later, "
            "otherwise empty. commitments = promises/things the persona said "
            "she would do (content = what was promised, about_username = who "
            "it was promised to, follow_up_question = a short casual way to "
            "check up on it later). stances = only strong opinions the "
            "persona actually stated (topic = short lowercase topic key, "
            "stance = one-sentence opinion in her voice). "
            "Return JSON exactly matching the schema."
        )

        try:
            outcome = await llm_provider.generate_structured(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                ExtractionOutcome,
                temperature=0.2,
                max_tokens=2000,
            )
        except Exception as e:
            logger.error("memory_extraction_failed", error=str(e))
            return None

        # Resolve usernames -> internal ids locally (never trust LLM with ids)
        username_to_id = {str(v).lstrip("@").lower(): k for k, v in usernames_by_id.items()}
        for item in outcome.memories:
            ids = []
            for name in item.about_usernames:
                normalized = str(name).lstrip("@").lower()
                if normalized in username_to_id:
                    ids.append(username_to_id[normalized])
            item.people_involved = ids
            # The memory's OWNER is the person it is primarily about — NOT
            # an arbitrary first-mentioned name. A memory about Alex stays
            # Alex's even if Sarah mentioned him first.
            primary = str(item.about_username or "").lstrip("@").lower()
            item.user_id = username_to_id.get(primary) or (ids[0] if ids else None)

        return outcome


async def get_or_create_user(
    session: AsyncSession,
    telegram_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
) -> User:
    stmt = select(User).where(User.telegram_id == telegram_id)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
        )
        session.add(user)
        await session.flush()

        relationship = Relationship(user_id=user.id)
        session.add(relationship)
        await session.flush()

    else:
        updated = False
        if username and user.username != username:
            user.username = username
            updated = True
        if first_name and user.first_name != first_name:
            user.first_name = first_name
            updated = True
        if last_name and user.last_name != last_name:
            user.last_name = last_name
            updated = True
        if updated:
            user.updated_at = datetime.utcnow()
            await session.flush()

    return user


async def get_or_create_conversation(
    session: AsyncSession,
    chat_id: int,
    chat_type: str,
    title: Optional[str] = None,
) -> Conversation:
    from sqlalchemy.orm import selectinload
    stmt = select(Conversation).options(selectinload(Conversation.state)).where(Conversation.telegram_chat_id == chat_id)
    result = await session.execute(stmt)
    conversation = result.scalar_one_or_none()

    if conversation is None:
        conversation = Conversation(
            telegram_chat_id=chat_id,
            chat_type=chat_type,
            title=title,
        )
        session.add(conversation)
        await session.flush()

        state = ConversationStateModel(
            conversation_id=conversation.id,
            state=ConversationState.AWAY,
        )
        session.add(state)
        await session.flush()
        # Populate the in-memory relationship so callers can access .state
        # without triggering an async lazy-load (MissingGreenlet).
        conversation.state = state

    elif title and conversation.title != title:
        conversation.title = title
        conversation.updated_at = datetime.utcnow()
        await session.flush()

    return conversation


async def ensure_emotional_state(session: AsyncSession) -> EmotionalState:
    stmt = select(EmotionalState)
    result = await session.execute(stmt)
    state = result.scalar_one_or_none()

    if state is None:
        state = EmotionalState()
        session.add(state)
        await session.flush()

    return state