import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import (
    User, Conversation, Message, Memory, MemoryType,
    Relationship, ConversationStateModel, ConversationState,
    EmotionalState, Base
)
from app.memory.store import get_or_create_user, get_or_create_conversation, MemoryStore, MemoryFormation
from app.memory.retrieval import MemoryRetriever, RetrievalContext, build_retrieval_context
from app.state.manager import ConversationStateManager, EmotionalStateManager, RelationshipManager
from app.social.decision import SocialDecisionEngine, ParticipationFactors, SocialScorer
from app.agent.core import Agent
from app.personality import PersonalityLoader, get_personality
from app.config import settings


@pytest.fixture
async def session():
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from app.config import settings

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def personality():
    return get_personality()


class TestDatabaseModels:
    @pytest.mark.asyncio
    async def test_create_user(self, session):
        user = await get_or_create_user(session, 12345, "testuser", "Test", "User")
        await session.commit()

        assert user.telegram_id == 12345
        assert user.username == "testuser"
        assert user.first_name == "Test"

    @pytest.mark.asyncio
    async def test_create_conversation(self, session):
        conv = await get_or_create_conversation(session, -100123, "group", "Test Group")
        await session.commit()

        assert conv.telegram_chat_id == -100123
        assert conv.chat_type == "group"
        assert conv.title == "Test Group"
        assert conv.state is not None
        assert conv.state.state == ConversationState.AWAY


class TestMemoryStore:
    @pytest.mark.asyncio
    async def test_create_memory(self, session):
        user = await get_or_create_user(session, 12345, "testuser")
        conv = await get_or_create_conversation(session, -100123, "group")
        await session.commit()

        store = MemoryStore(session)
        memory = await store.create_memory(
            memory_type=MemoryType.EPISODIC,
            content="User said they love Python",
            user_id=user.id,
            conversation_id=conv.id,
            people_involved=[12345],
            topics=["python", "programming"],
            importance=0.8,
            confidence=0.9,
        )
        await session.commit()

        assert memory.id is not None
        assert memory.memory_type == MemoryType.EPISODIC
        assert memory.importance == 0.8
        assert memory.embedding is not None
        assert isinstance(memory.embedding, list) and len(memory.embedding) > 0

    @pytest.mark.asyncio
    async def test_retrieve_memories(self, session):
        user = await get_or_create_user(session, 12345, "testuser")
        conv = await get_or_create_conversation(session, -100123, "group")
        await session.commit()

        store = MemoryStore(session)
        await store.create_memory(
            memory_type=MemoryType.SEMANTIC,
            content="User loves Python programming",
            user_id=user.id,
            conversation_id=conv.id,
            people_involved=[12345],
            topics=["python"],
            importance=0.7,
        )
        await store.create_memory(
            memory_type=MemoryType.PREFERENCE,
            content="User dislikes Java",
            user_id=user.id,
            conversation_id=conv.id,
            people_involved=[12345],
            topics=["java"],
            importance=0.6,
        )
        await session.commit()

        memories = await store.retrieve_memories(
            query_text="What does the user think about programming languages?",
            user_id=user.id,
            conversation_id=conv.id,
            limit=5,
        )

        assert len(memories) == 2
        assert all(m[0].user_id == user.id for m in memories)


class TestMemoryRetrieval:
    @pytest.mark.asyncio
    async def test_retrieval_context_building(self, session):
        user = await get_or_create_user(session, 12345, "testuser")
        conv = await get_or_create_conversation(session, -100123, "group")
        await session.commit()

        context = await build_retrieval_context(
            session,
            conv.id,
            user.id,
            "I love Python!",
            [
                {"user_id": user.id, "text": "Python is great", "timestamp": datetime.utcnow()},
            ],
        )

        assert context.query_text == "I love Python!"
        assert context.conversation_id == conv.id
        assert context.user_id == user.id
        from app.config import settings as _settings
        assert context.max_results == _settings.max_memory_results


class TestConversationState:
    @pytest.mark.asyncio
    async def test_state_transitions(self, session):
        conv = await get_or_create_conversation(session, -100123, "group")
        await session.commit()

        manager = ConversationStateManager(session)

        await manager.update_state(conv.id, new_state=ConversationState.OBSERVING)
        await session.commit()

        ctx = await manager.get_context(conv.id)
        assert ctx.state.state == ConversationState.OBSERVING

        await manager.update_state(conv.id, new_state=ConversationState.PARTICIPATING, is_participating=True)
        await session.commit()

        ctx = await manager.get_context(conv.id)
        assert ctx.state.state == ConversationState.PARTICIPATING
        assert ctx.state.is_participating is True


class TestEmotionalState:
    @pytest.mark.asyncio
    async def test_emotional_updates(self, session):
        manager = EmotionalStateManager(session)

        await manager.update_from_event("funny", intensity=1.0)
        await session.commit()

        state = await manager.get_state()
        assert state["amusement"] > 0
        assert state["mood"] > 0

        await manager.update_from_event("insult", intensity=1.0)
        await session.commit()

        state = await manager.get_state()
        assert state["annoyance"] > 0
        # Insult pushes mood downward and raises annoyance.
        assert state["mood"] < 0
        assert state["annoyance"] > 0.0


class TestRelationshipManager:
    @pytest.mark.asyncio
    async def test_relationship_updates(self, session):
        user = await get_or_create_user(session, 12345, "testuser")
        await session.commit()

        manager = RelationshipManager(session)

        await manager.update_from_interaction(user.id, "positive", 1.0)
        await session.commit()

        rel = await manager.get_relationship_summary(user.id)
        assert rel["affection"] > 0
        assert rel["trust"] > 0

        await manager.update_from_interaction(user.id, "negative", 1.0)
        await session.commit()

        rel = await manager.get_relationship_summary(user.id)
        assert rel["affection"] < 0.05
        assert rel["annoyance"] > 0


class TestSocialDecisionEngine:
    def test_participation_scoring(self):
        engine = SocialDecisionEngine()

        factors = ParticipationFactors(
            direct_mention=1.0,
            topic_relevance=0.8,
            personal_interest=0.6,
            relationship_relevance=0.5,
            conversation_energy=0.7,
            curiosity=0.6,
            current_mood=0.5,
            recent_activity=0.3,
            conversation_saturation=0.2,
        )

        score = engine.calculate_participation_score(factors)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_no_participation_without_reason(self):
        engine = SocialDecisionEngine()

        factors = ParticipationFactors(
            direct_mention=0.0,
            topic_relevance=0.1,
            personal_interest=0.1,
            relationship_relevance=0.1,
            conversation_energy=0.2,
            curiosity=0.2,
            current_mood=0.3,
            recent_activity=0.5,
            conversation_saturation=0.8,
        )

        score = engine.calculate_participation_score(factors)
        assert score < 0.4

    def test_direct_mention_boost(self):
        engine = SocialDecisionEngine()

        factors_with = ParticipationFactors(direct_mention=1.0)
        factors_without = ParticipationFactors(direct_mention=0.0)

        score_with = engine.calculate_participation_score(factors_with)
        score_without = engine.calculate_participation_score(factors_without)

        assert score_with > score_without

    def test_response_delay_calculation(self):
        engine = SocialDecisionEngine()

        with patch("app.social.decision.random.uniform", return_value=0.0):
            direct_delay = engine.calculate_response_delay(is_direct_mention=True, is_autonomous=False)
            autonomous_delay = engine.calculate_response_delay(is_direct_mention=False, is_autonomous=True)
            normal_delay = engine.calculate_response_delay(is_direct_mention=False, is_autonomous=False)

        assert direct_delay < autonomous_delay
        assert settings.min_response_delay <= direct_delay <= settings.max_response_delay


class TestAgent:
    @pytest.mark.asyncio
    async def test_agent_process_message(self, session):
        user = await get_or_create_user(session, 12345, "testuser")
        conv = await get_or_create_conversation(session, -100123, "group")
        await session.commit()

        with patch("app.agent.core.llm_provider") as mock_llm:
            from app.social.decision import SocialDecision, SocialAction
            mock_llm.generate_structured = AsyncMock(return_value=SocialDecision(
                action=SocialAction.REPLY,
                reasoning_summary="Test response",
                current_topic="testing",
                confidence=0.8,
                response="Hello there!",
                continue_participating=True,
            ))

            agent = Agent(session, "testbot")
            decision = await agent.process_message(
                conversation_id=conv.id,
                user=user,
                message_text="Hey bot!",
                is_direct_mention=True,
            )

            assert decision.action == SocialAction.REPLY
            assert decision.response == "Hello there!"


class TestMemoryFormation:
    @pytest.mark.asyncio
    async def test_memory_extraction_skips_trivial(self, session):
        user = await get_or_create_user(session, 12345, "testuser")
        conv = await get_or_create_conversation(session, -100123, "group")
        await session.commit()

        from pydantic import BaseModel

        class MockOutcome(BaseModel):
            should_remember: bool = False
            memories: list = []

        with patch("app.llm.llm_provider") as mock_llm:
            mock_llm.generate_structured = AsyncMock(return_value=MockOutcome())

            formation = MemoryFormation(session)
            memories = await formation.extract_from_conversation(conv.id, [
                {"user_id": user.id, "text": "lol", "username": "testuser"},
            ])

            assert len(memories) == 0


class TestPersonality:
    def test_personality_loaded(self):
        personality = get_personality()
        assert personality.character["name"] == "Airi"
        assert "openness" in personality.personality_traits
        assert len(personality.behavioral_tendencies) > 0
        assert len(personality.character_boundaries) > 0

    def test_speech_style_values(self):
        personality = get_personality()
        speech = personality.speech_style
        assert 0 <= speech.formality <= 1
        assert 0 <= speech.verbosity <= 1
        assert 0 <= speech.humor <= 1


class TestTopicDriftPrevention:
    @pytest.mark.asyncio
    async def test_memory_relevance_scoring(self, session):
        user = await get_or_create_user(session, 12345, "testuser")
        conv = await get_or_create_conversation(session, -100123, "group")
        await session.commit()

        store = MemoryStore(session)
        await store.create_memory(
            memory_type=MemoryType.EPISODIC,
            content="User talked about cooking pasta yesterday",
            user_id=user.id,
            conversation_id=conv.id,
            people_involved=[12345],
            topics=["cooking", "pasta"],
            importance=0.6,
        )
        await session.commit()

        retriever = MemoryRetriever(session)
        context = RetrievalContext(
            query_text="What do you think about Rust programming?",
            conversation_id=conv.id,
            user_id=user.id,
            current_topic="programming languages",
        )

        results = await retriever.retrieve(context)

        assert len(results) == 0 or all(r.relevance_score < 0.5 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])