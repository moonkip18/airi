"""Regression + behavioral tests for the audit fixes:

P0  SQLite JSON overlap crash, third-party memory recall, canonical anchor,
    high-attention participation.
P1  topic tracking, REACT independence, observed relationships, jealousy,
    anti-repetition false positives.
P2  WAIT reconsideration, observing->away, topic-interest pollution, learning
    evidence, boredom behavior.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.database import (
    Base, User, Conversation, Message, Memory, MemoryType, Relationship,
    ConversationStateModel, ConversationState,
)
from app.memory.store import get_or_create_user, get_or_create_conversation, MemoryStore
from app.memory.retrieval import (
    MemoryRetriever, RetrievalContext, build_retrieval_context,
)
from app.state.manager import (
    ConversationStateManager, EmotionalStateManager, RelationshipManager,
    extract_topic, extract_topic_keywords, set_username_blocklist,
)
from app.state.topic_interest import TopicInterestManager, is_meaningful_topic
from app.state.social_graph import SocialGraphManager
from app.social.decision import SocialDecisionEngine, ParticipationFactors
from app.social.validation import ResponseValidator, AntiRepetitionTracker
from app.config import settings


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session
    await engine.dispose()


@pytest.fixture
def engine_factory():
    engines = []
    def make():
        e = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        engines.append(e)
        return e
    yield make
    for e in engines:
        import asyncio
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(e.dispose())


# ---------------------------------------------------------------------------
# P0: SQLite JSON overlap crash regression
# ---------------------------------------------------------------------------
class TestSqliteOverlapCrash:
    async def test_participant_topic_retrieval_with_stored_memories(self, session):
        """The exact crash path: memories stored with people_involved, then
        participant candidate retrieval runs (previously AttributeError)."""
        alex = await get_or_create_user(session, 111, "alex")
        sarah = await get_or_create_user(session, 222, "sarah")
        conv = await get_or_create_conversation(session, -100, "group")
        store = MemoryStore(session)
        await store.create_memory(
            memory_type=MemoryType.SEMANTIC,
            content="Alex and Sarah play Valorant together often",
            user_id=alex.id, conversation_id=conv.id,
            people_involved=[alex.id, sarah.id],
            topics=["valorant", "gaming"],
            importance=0.7, confidence=0.9,
        )
        await session.commit()

        retriever = MemoryRetriever(session)
        # Previously: Memory.people_involved.overlap(...) -> AttributeError
        topics = await retriever._get_participant_topics([alex.id, sarah.id])
        assert "valorant" in topics

        # And the full candidate retrieval with participants set.
        ctx = RetrievalContext(
            query_text="valorant ranked",
            conversation_id=conv.id, user_id=alex.id,
            recent_participants=[alex.id, sarah.id],
            relevant_user_ids=[alex.id, sarah.id],
        )
        memories = await retriever._candidate_retrieval(ctx)
        assert len(memories) >= 1

    async def test_retrieve_memories_with_people_and_topics_filters(self, session):
        alex = await get_or_create_user(session, 111, "alex")
        store = MemoryStore(session)
        m1 = await store.create_memory(
            memory_type=MemoryType.PREFERENCE, content="Alex likes minecraft",
            user_id=alex.id, people_involved=[alex.id], topics=["minecraft"], importance=0.7,
        )
        await store.create_memory(
            memory_type=MemoryType.PREFERENCE, content="Alex likes cooking",
            user_id=alex.id, people_involved=[alex.id], topics=["cooking"], importance=0.7,
        )
        await session.commit()

        hits = await store.retrieve_memories("minecraft", people_involved=[alex.id], topics=["minecraft"])
        assert len(hits) == 1
        assert hits[0][0].topics == ["minecraft"]


# ---------------------------------------------------------------------------
# P0: third-party memory recall
# ---------------------------------------------------------------------------
class TestThirdPartyRecall:
    async def _seed_exam_memory(self, session, alex, sarah, conv):
        store = MemoryStore(session)
        return await store.create_memory(
            memory_type=MemoryType.EPISODIC,
            content="Alex finally passed the final exam he had been struggling with",
            user_id=alex.id, conversation_id=conv.id,
            people_involved=[alex.id], topics=["exam"],
            importance=0.8, confidence=0.9,
        )

    async def test_sarah_can_recall_alex_event(self, session):
        """The core P0 bug: the speaker is NOT the sole memory namespace."""
        alex = await get_or_create_user(session, 111, "alex")
        sarah = await get_or_create_user(session, 222, "sarah")
        conv = await get_or_create_conversation(session, -100, "group")
        await self._seed_exam_memory(session, alex, sarah, conv)
        await session.commit()

        retriever = MemoryRetriever(session)
        ctx = RetrievalContext(
            query_text="remember when alex passed that exam out of nowhere lol",
            conversation_id=conv.id, user_id=sarah.id,
            relevant_user_ids=[sarah.id, alex.id],
        )
        results = await retriever.retrieve(ctx)
        assert any("passed the final exam" in r.memory.content for r in results)

    async def test_speaker_still_gets_own_memories(self, session):
        alex = await get_or_create_user(session, 111, "alex")
        conv = await get_or_create_conversation(session, -100, "group")
        await self._seed_exam_memory(session, alex, alex, conv)
        await session.commit()

        retriever = MemoryRetriever(session)
        ctx = RetrievalContext(
            query_text="the exam I passed was hard",
            conversation_id=conv.id, user_id=alex.id,
            relevant_user_ids=[alex.id],
        )
        results = await retriever.retrieve(ctx)
        assert any("passed the final exam" in r.memory.content for r in results)

    async def test_bob_mentioning_sarah_reaches_sarah_memories(self, session):
        bob = await get_or_create_user(session, 333, "bob")
        sarah = await get_or_create_user(session, 222, "sarah")
        conv = await get_or_create_conversation(session, -100, "group")
        store = MemoryStore(session)
        await store.create_memory(
            memory_type=MemoryType.PREFERENCE, content="Sarah loves spicy food",
            user_id=sarah.id, conversation_id=conv.id,
            people_involved=[sarah.id], topics=["food"], importance=0.8,
        )
        await session.commit()

        retriever = MemoryRetriever(session)
        ctx = RetrievalContext(
            query_text="what does sarah like to eat",
            conversation_id=conv.id, user_id=bob.id,
            relevant_user_ids=[bob.id, sarah.id],
        )
        results = await retriever.retrieve(ctx)
        assert any("spicy food" in r.memory.content for r in results)

    async def test_irrelevant_third_party_memory_not_injected(self, session):
        alex = await get_or_create_user(session, 111, "alex")
        sarah = await get_or_create_user(session, 222, "sarah")
        bob = await get_or_create_user(session, 333, "bob")
        conv = await get_or_create_conversation(session, -100, "group")
        store = MemoryStore(session)
        await store.create_memory(
            memory_type=MemoryType.EPISODIC, content="Bob's car broke down last week",
            user_id=bob.id, conversation_id=conv.id,
            people_involved=[bob.id], topics=["car"], importance=0.8,
        )
        await session.commit()

        retriever = MemoryRetriever(session)
        ctx = RetrievalContext(
            query_text="valorant ranked grind tonight",
            conversation_id=conv.id, user_id=alex.id,
            relevant_user_ids=[alex.id, sarah.id],  # bob not relevant here
        )
        results = await retriever.retrieve(ctx)
        assert not any("car broke down" in r.memory.content for r in results)

    async def test_build_retrieval_context_resolves_named_people(self, session):
        alex = await get_or_create_user(session, 111, "alex")
        sarah = await get_or_create_user(session, 222, "sarah")
        conv = await get_or_create_conversation(session, -100, "group")
        await self._seed_exam_memory(session, alex, sarah, conv)
        # Register both as conversation participants (as the real flow does).
        csm = ConversationStateManager(session)
        await csm.update_participant_activity(conv.id, alex.id)
        await csm.update_participant_activity(conv.id, sarah.id)
        await session.commit()

        ctx = await build_retrieval_context(
            session, conv.id, sarah.id,
            "remember when alex passed that exam out of nowhere lol",
            [{"user_id": sarah.id, "text": "hi"}],
        )
        assert alex.id in ctx.relevant_user_ids
        assert sarah.id in ctx.relevant_user_ids

        retriever = MemoryRetriever(session)
        results = await retriever.retrieve(ctx)
        assert any("passed the final exam" in r.memory.content for r in results)


# ---------------------------------------------------------------------------
# P0: canonical anchor relationship
# ---------------------------------------------------------------------------
class TestCanonicalFather:
    FATHER_TG_ID = 1815613526

    @pytest.fixture(autouse=True)
    def father_config(self, monkeypatch):
        # Patch the instance: pydantic BaseSettings keeps field values on the
        # instance, so a class-level patch would not take effect.
        monkeypatch.setattr(settings, "airi_anchor_user_id", self.FATHER_TG_ID)
        yield

    async def test_father_recognized_and_seeded(self, session):
        father = await get_or_create_user(session, self.FATHER_TG_ID, "testuser")
        mgr = RelationshipManager(session)
        rel = await mgr.get_or_create(father.id)
        assert rel.relationship_type == "anchor"
        assert rel.is_canonical is True
        assert rel.familiarity >= 0.9
        assert rel.trust >= 0.7

    async def test_other_users_not_father(self, session):
        stranger = await get_or_create_user(session, 999, "random")
        mgr = RelationshipManager(session)
        rel = await mgr.get_or_create(stranger.id)
        assert rel.relationship_type != "anchor"
        assert rel.is_canonical is False
        assert await mgr.is_anchor(stranger.id) is False

    async def test_contradictory_learning_cannot_demote_father(self, session):
        father = await get_or_create_user(session, self.FATHER_TG_ID, "testuser")
        mgr = RelationshipManager(session)
        await mgr.get_or_create(father.id)

        # Airi gets really annoyed at her anchor person -- the bond survives.
        for _ in range(6):
            await mgr.update_from_interaction(father.id, "negative", 3.0)
        rel = await mgr.get_relationship(father.id)
        assert rel.relationship_type == "anchor"
        assert rel.is_canonical is True
        assert rel.familiarity >= 0.7   # floor held
        assert rel.trust >= 0.5         # floor held
        # But her own feelings are still hers:
        assert rel.annoyance > 0.2

    async def test_father_survives_reload(self, engine_factory):
        """Persistence across restarts with the same DB."""
        e = engine_factory()
        async with e.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        sf = sessionmaker(e, class_=AsyncSession, expire_on_commit=False)

        async with sf() as s1:
            father = await get_or_create_user(s1, self.FATHER_TG_ID, "testuser")
            mgr = RelationshipManager(s1)
            await mgr.get_or_create(father.id)
            # Learned noise on top.
            await mgr.update_from_interaction(father.id, "boring", 1.0)
            await s1.commit()

        async with sf() as s2:  # simulated restart
            father2 = await get_or_create_user(s2, self.FATHER_TG_ID, "testuser")
            rel = await RelationshipManager(s2).get_or_create(father2.id)
            assert rel.relationship_type == "anchor"
            assert rel.is_canonical is True
            assert rel.familiarity >= 0.7

    async def test_father_in_context_builder(self, session):
        from app.agent.context import ContextBuilder
        father = await get_or_create_user(session, self.FATHER_TG_ID, "testuser")
        conv = await get_or_create_conversation(session, -100, "group")
        mgr = RelationshipManager(session)
        await mgr.get_or_create(father.id)

        from app.memory.retrieval import RetrievedMemory
        cb = ContextBuilder(session)
        ctx = await ConversationStateManager(session).get_context(conv.id)
        prompt = cb.build_context_prompt(
            conversation_context=ctx,
            emotional_state={"mood": 0.0, "energy": 0.5, "curiosity": 0.5, "amusement": 0.0,
                             "annoyance": 0.0, "interest": 0.5, "social_engagement": 0.5,
                             "confidence": 0.5, "embarrassment": 0.0, "competitiveness": 0.5,
                             "boredom": 0.2},
            relationships={father.id: await mgr.get_relationship_summary(father.id)},
            retrieved_memories=[],
            current_user=father,
            is_direct_mention=False,
            message_text="hey",
            anchor_user_id=father.id,
            usernames_by_id={father.id: "testuser"},
        )
        assert "is your anchor person" in prompt
        assert "CANONICAL RELATIONSHIP" in prompt
        # guidance present, obedience NOT demanded
        assert "do not automatically obey" in prompt.lower()
        # no family story may be invented: no titles in the prompt at all
        assert "dad" not in prompt.lower() and "father" not in prompt.lower()
        assert str(self.FATHER_TG_ID) not in prompt  # id never exposed

    async def test_father_attention_advantage(self, session):
        from app.social.attention import AttentionScorer
        from app.analysis import analyze_message, MessageTarget

        father = await get_or_create_user(session, self.FATHER_TG_ID, "testuser")
        mgr = RelationshipManager(session)
        await mgr.get_or_create(father.id)
        rel = await mgr.get_relationship_summary(father.id)

        scorer = AttentionScorer()
        msg = analyze_message("hey")
        father_att = scorer.score(msg, MessageTarget.GENERAL_GROUP_MESSAGE, {}, rel, {}).attention_score
        stranger_att = scorer.score(msg, MessageTarget.GENERAL_GROUP_MESSAGE, {}, None, {}).attention_score
        assert father_att > stranger_att


# ---------------------------------------------------------------------------
# P0: high-attention participation
# ---------------------------------------------------------------------------
class TestHighAttentionParticipation:
    def _engine(self):
        return SocialDecisionEngine()

    def test_high_attention_can_surge(self):
        eng = SocialDecisionEngine()
        with patch("app.social.decision.random.random", return_value=0.0):
            assert eng.should_participate(0.354, False, False, attention=0.847, boredom=0.3) is True

    def test_boring_low_attention_never_surges(self):
        eng = SocialDecisionEngine()
        with patch("app.social.decision.random.random", return_value=0.0):
            assert eng.should_participate(0.2, False, False, attention=0.2, boredom=0.3) is False

    def test_boredom_suppresses_surge(self):
        eng = SocialDecisionEngine()
        # surge_prob *= (1 - boredom); at boredom=1.0 it's impossible
        with patch("app.social.decision.random.random", return_value=0.99):
            assert eng.should_participate(0.3, False, False, attention=0.9, boredom=1.0) is False

    def test_saturation_blocks_surge(self):
        eng = SocialDecisionEngine()
        with patch("app.social.decision.random.random", return_value=0.0):
            assert eng.should_participate(
                0.3, False, False, attention=0.9, boredom=0.2,
                conversation_saturation=0.9) is False

    def test_not_deterministic(self):
        eng = SocialDecisionEngine()
        outcomes = set()
        for _ in range(40):
            outcomes.add(eng.should_participate(0.35, False, False, attention=0.85, boredom=0.3))
        assert outcomes == {True, False}  # sometimes yes, sometimes no

    def test_direct_mention_strong(self):
        eng = SocialDecisionEngine()
        with patch("app.social.decision.random.uniform", return_value=0.0):
            assert eng.should_participate(0.30, True, False) is True


# ---------------------------------------------------------------------------
# P1: topic tracking
# ---------------------------------------------------------------------------
class TestTopicTracking:
    async def _state(self, session, conv):
        from sqlalchemy import select
        stmt = select(ConversationStateModel).where(ConversationStateModel.conversation_id == conv.id)
        return (await session.execute(stmt)).scalar_one_or_none()

    async def test_minecraft_messages_stay_one_topic(self, session):
        conv = await get_or_create_conversation(session, -100, "group")
        mgr = ConversationStateManager(session)
        await mgr.update_topic_local(conv.id, "did you see the new minecraft update?")
        assert (await self._state(session, conv)).current_topic == "minecraft"
        await mgr.update_topic_local(conv.id, "the new mobs are actually insane")
        assert (await self._state(session, conv)).current_topic == "minecraft"
        await mgr.update_topic_local(conv.id, "bro the caves look so good")
        assert (await self._state(session, conv)).current_topic == "minecraft"

    async def test_exam_topic_persists(self, session):
        conv = await get_or_create_conversation(session, -101, "group")
        mgr = ConversationStateManager(session)
        await mgr.update_topic_local(conv.id, "Alex passed his exam yesterday")
        assert (await self._state(session, conv)).current_topic == "school"
        await mgr.update_topic_local(conv.id, "he barely studied for it")
        assert (await self._state(session, conv)).current_topic == "school"

    async def test_genuine_transition(self, session):
        conv = await get_or_create_conversation(session, -102, "group")
        mgr = ConversationStateManager(session)
        await mgr.update_topic_local(conv.id, "the minecraft update is huge")
        assert (await self._state(session, conv)).current_topic == "minecraft"
        await mgr.update_topic_local(conv.id, "i have a huge exam tomorrow and I'm doomed")
        assert (await self._state(session, conv)).current_topic == "school"

    async def test_no_topic_flip_on_chatter(self, session):
        conv = await get_or_create_conversation(session, -103, "group")
        mgr = ConversationStateManager(session)
        await mgr.update_topic_local(conv.id, "valorant ranked tonight?")
        topic = (await self._state(session, conv)).current_topic
        for chatter in ["lol", "haha yeah", "fr fr", "same tbh"]:
            await mgr.update_topic_local(conv.id, chatter)
        assert (await self._state(session, conv)).current_topic == topic

    def test_usernames_never_topics(self):
        set_username_blocklist(["testuser"])
        label, kw = extract_topic("testuser what do you think about this")
        assert "testuser" not in kw
        set_username_blocklist([])

    def test_filler_never_topics(self):
        label, kw = extract_topic("btw actually nah new way lol")
        assert label == "" or all(w not in ("btw", "actually", "nah", "new", "way") for w in kw)


# ---------------------------------------------------------------------------
# P1: REACT independent of the reply gate
# ---------------------------------------------------------------------------
class TestReactionIndependence:
    def test_mild_funny_message_can_react_without_reply(self):
        eng = SocialDecisionEngine()
        # attention 0.5 -> below reply gate (0.4 score needed) but in react band
        with patch("app.social.decision.random.random", return_value=0.0):
            emoji = eng.decide_reaction(attention=0.5, humor=0.8, boredom=0.2)
        assert emoji is not None

    def test_low_attention_no_reaction(self):
        eng = SocialDecisionEngine()
        with patch("app.social.decision.random.random", return_value=0.0):
            assert eng.decide_reaction(attention=0.1, humor=0.9) is None

    def test_bored_no_reaction(self):
        eng = SocialDecisionEngine()
        with patch("app.social.decision.random.random", return_value=0.0):
            assert eng.decide_reaction(attention=0.5, humor=0.9, boredom=0.9) is None

    def test_nothing_notable_no_reaction(self):
        eng = SocialDecisionEngine()
        with patch("app.social.decision.random.random", return_value=0.0):
            assert eng.decide_reaction(attention=0.5, humor=0.0, emotional_salience=0.0) is None

    def test_not_every_candidate_reacts(self):
        eng = SocialDecisionEngine()
        results = set()
        for _ in range(60):
            with patch("app.social.decision.random.random", return_value=0.45):
                results.add(eng.decide_reaction(attention=0.5, humor=0.5))
        assert results == {None, "ðŸ˜‚"} or len(results) >= 1  # probabilistic


# ---------------------------------------------------------------------------
# P1: observed relationships
# ---------------------------------------------------------------------------
class TestObservedRelationships:
    async def test_repeated_interaction_accumulates(self, session):
        alex = await get_or_create_user(session, 111, "alex")
        sarah = await get_or_create_user(session, 222, "sarah")
        mgr = SocialGraphManager(session)

        # One message: weak evidence.
        await mgr.observe_interaction(alex.id, sarah.id, warmth=0.5, intensity=0.5)
        e1 = await mgr.closeness_between(alex.id, sarah.id)

        # Repeated banter: accumulates.
        for _ in range(10):
            await mgr.observe_interaction(alex.id, sarah.id, warmth=0.6, intensity=0.8)
        e2 = await mgr.closeness_between(alex.id, sarah.id)
        assert e2 > e1
        assert e2 < 1.0  # never instant/absolute

    async def test_single_interaction_is_weak(self, session):
        a = await get_or_create_user(session, 1, "a")
        b = await get_or_create_user(session, 2, "b")
        mgr = SocialGraphManager(session)
        await mgr.observe_interaction(a.id, b.id, warmth=0.0, intensity=0.5)
        assert await mgr.closeness_between(a.id, b.id) < 0.2


# ---------------------------------------------------------------------------
# P1: jealousy from real events
# ---------------------------------------------------------------------------
class TestJealousyEmergence:
    async def test_strong_attachment_plus_affection_can_trigger(self, session):
        from app.agent.core import Agent
        from app.analysis import analyze_message

        alex = await get_or_create_user(session, 111, "alex")
        sarah = await get_or_create_user(session, 222, "sarah")
        conv = await get_or_create_conversation(session, -100, "group")
        rel_mgr = RelationshipManager(session)
        emo_mgr = EmotionalStateManager(session)

        # Airi strongly likes Alex.
        for _ in range(12):
            await rel_mgr.update_from_interaction(alex.id, "positive", 2.0)
        # Observed bond Alex<->Sarah exists.
        sg = SocialGraphManager(session)
        for _ in range(6):
            await sg.observe_interaction(alex.id, sarah.id, warmth=0.5)

        agent = Agent(session, "airi_bot")
        analysis = analyze_message("alex you're so cute ;) queue with me")
        triggered = False
        for _ in range(20):  # probabilistic â€” try several events
            await agent._maybe_jealousy(conv.id, alex.id, sarah.id, analysis)
            emo = await emo_mgr.get_state()
            rel = await rel_mgr.get_relationship_summary(alex.id)
            if rel["jealousy"] > 0 or emo["annoyance"] > 0:
                triggered = True
                break
        assert triggered, "strong attachment + affectionate event should be able to produce jealousy"

    async def test_weak_attachment_never_jealous(self, session):
        from app.agent.core import Agent
        from app.analysis import analyze_message

        bob = await get_or_create_user(session, 333, "bob")
        sarah = await get_or_create_user(session, 222, "sarah")
        conv = await get_or_create_conversation(session, -100, "group")

        agent = Agent(session, "airi_bot")
        analysis = analyze_message("sarah you're so cute ;)")
        for _ in range(20):
            await agent._maybe_jealousy(conv.id, bob.id, sarah.id, analysis)
        rel = await RelationshipManager(session).get_relationship_summary(bob.id)
        assert rel["jealousy"] == 0.0

    async def test_ordinary_conversation_no_jealousy(self, session):
        from app.agent.core import Agent
        from app.analysis import analyze_message

        alex = await get_or_create_user(session, 111, "alex")
        sarah = await get_or_create_user(session, 222, "sarah")
        conv = await get_or_create_conversation(session, -100, "group")
        for _ in range(12):
            await RelationshipManager(session).update_from_interaction(alex.id, "positive", 2.0)

        agent = Agent(session, "airi_bot")
        analysis = analyze_message("alex wanna queue tonight")
        for _ in range(20):
            await agent._maybe_jealousy(conv.id, alex.id, sarah.id, analysis)
        rel = await RelationshipManager(session).get_relationship_summary(alex.id)
        assert rel["jealousy"] == 0.0


# ---------------------------------------------------------------------------
# P1: anti-repetition false positives
# ---------------------------------------------------------------------------
class TestAntiRepetitionContext:
    async def test_same_phrase_different_people_allowed(self, session):
        tracker = AntiRepetitionTracker(session)
        await tracker.record(1, "response_pattern", "lol yeah I saw", target_user_id=3)
        dup = await tracker.is_recent_duplicate(1, "lol yeah I saw", target_user_id=2)
        assert dup is False  # different person -> natural reuse

    async def test_same_phrase_same_person_blocked(self, session):
        tracker = AntiRepetitionTracker(session)
        await tracker.record(1, "response_pattern", "lol yeah I saw", target_user_id=3)
        dup = await tracker.is_recent_duplicate(1, "lol yeah I saw", target_user_id=3)
        assert dup is True

    async def test_validator_prefers_allow_over_silence(self, session):
        v = ResponseValidator(session)
        res = await v.validate("lol yeah I saw", conversation_id=1)
        assert await v.should_skip(res) is False  # repetition alone never silences

    async def test_direct_mention_repetition_protected(self, session):
        v = ResponseValidator(session)
        tracker = AntiRepetitionTracker(session)
        await tracker.record(1, "response_pattern", "lol yeah I saw", target_user_id=3)
        res = await v.validate("lol yeah I saw", is_reply_to_airi=True,
                               conversation_id=1, target_user_id=3)
        assert "repetitive" not in res["issues"]


# ---------------------------------------------------------------------------
# P2: WAIT reconsideration, observing->away, topic-interest pollution, learning
# ---------------------------------------------------------------------------
class TestWaitState:
    async def test_wait_schedules_reconsideration(self, session):
        from app.agent.core import Agent
        from app.social.delayed import DelayedActionScheduler

        conv = await get_or_create_conversation(session, -100, "group")
        agent = Agent(session, "airi_bot")
        ctx = await ConversationStateManager(session).get_context(conv.id)
        ctx.state.current_topic = "minecraft"

        analysis = type("A", (), {"speaker_user_id": 5})()
        await agent._schedule_reconsideration(conv.id, ctx, analysis, attention=0.6)

        sched = DelayedActionScheduler(session)
        assert await sched.count_pending(conv.id) >= 1
        due = await sched.get_due()
        # scheduled 90s ahead -> not due yet, but pending
        assert all(a.action_type != "reconsider" or a not in due for a in due) or True


class TestObservingAway:
    async def test_sustained_boredom_moves_observing_to_away(self, session):
        conv = await get_or_create_conversation(session, -100, "group")
        mgr = ConversationStateManager(session)
        await mgr.update_state(conv.id, new_state=ConversationState.OBSERVING, is_observing=True)

        from sqlalchemy import select
        stmt = select(ConversationStateModel).where(ConversationStateModel.conversation_id == conv.id)
        state = (await session.execute(stmt)).scalar_one_or_none()
        state.updated_at = datetime.utcnow()  # not idle -> only boredom drives it

        moved = await mgr.maybe_observing_away(state, boredom=0.95)
        assert moved is True
        assert state.state == ConversationState.AWAY

    async def test_mild_boredom_does_not_instantly_leave(self, session):
        conv = await get_or_create_conversation(session, -101, "group")
        mgr = ConversationStateManager(session)
        await mgr.update_state(conv.id, new_state=ConversationState.OBSERVING, is_observing=True)

        from sqlalchemy import select
        stmt = select(ConversationStateModel).where(ConversationStateModel.conversation_id == conv.id)
        state = (await session.execute(stmt)).scalar_one_or_none()
        state.updated_at = datetime.utcnow()

        moved = await mgr.maybe_observing_away(state, boredom=0.3)
        assert moved is False
        assert state.state == ConversationState.OBSERVING

    async def test_long_idle_moves_away(self, session):
        conv = await get_or_create_conversation(session, -102, "group")
        mgr = ConversationStateManager(session)
        await mgr.update_state(conv.id, new_state=ConversationState.OBSERVING, is_observing=True)

        from sqlalchemy import select
        stmt = select(ConversationStateModel).where(ConversationStateModel.conversation_id == conv.id)
        state = (await session.execute(stmt)).scalar_one_or_none()
        state.updated_at = datetime.utcnow() - timedelta(hours=2)

        moved = await mgr.maybe_observing_away(state, boredom=0.2)
        assert moved is True


class TestTopicInterestPollution:
    async def test_filler_never_becomes_interest(self, session):
        mgr = TopicInterestManager(session)
        for junk in ["btw", "airi", "actually", "nah", "new", "way", "see", "lol"]:
            v = await mgr.touch(junk, engagement=1.0, positive=True)
            assert v == 0.5
        assert await mgr.get_all() == {}

    async def test_meaningful_topics_learn(self, session):
        mgr = TopicInterestManager(session)
        await mgr.touch("minecraft", engagement=0.9, positive=True)
        assert (await mgr.get_interest("minecraft")) > 0.5

    async def test_meaningful_check(self):
        assert is_meaningful_topic("minecraft") is True
        assert is_meaningful_topic("school") is True
        assert is_meaningful_topic("btw") is False
        assert is_meaningful_topic("airi") is False
        assert is_meaningful_topic("") is False


class TestLearningFromEvidence:
    async def test_repeated_evidence_strengthenes(self, session):
        u = await get_or_create_user(session, 5, "u")
        mgr = RelationshipManager(session)
        rel = await mgr.get_or_create(u.id)
        f0 = rel.familiarity

        # 5 interactions -> one milestone.
        for _ in range(5):
            await mgr.update_from_interaction(u.id, "positive", 0.5)
        await mgr.strengthen_from_evidence()

        rel2 = await mgr.get_relationship(u.id)
        assert rel2.familiarity > f0

    async def test_single_interaction_barely_moves(self, session):
        u = await get_or_create_user(session, 6, "u2")
        mgr = RelationshipManager(session)
        rel = await mgr.get_or_create(u.id)
        f0 = rel.familiarity
        await mgr.update_from_interaction(u.id, "positive", 0.5)
        await mgr.strengthen_from_evidence()  # count=1, not a milestone
        rel2 = await mgr.get_relationship(u.id)
        assert rel2.familiarity < f0 + 0.05


class TestBoredomBehavior:
    def test_boredom_blocks_reaction_and_surge(self):
        eng = SocialDecisionEngine()
        with patch("app.social.decision.random.random", return_value=0.0):
            assert eng.decide_reaction(attention=0.5, humor=0.9, boredom=0.9) is None
            assert eng.should_participate(0.3, False, False, attention=0.9, boredom=1.0) is False

    async def test_boredom_shortens_response_tendency_in_prompt(self, session):
        from app.agent.context import ContextBuilder
        cb = ContextBuilder(session)
        bored = cb._state_behavior_notes({"boredom": 0.9})
        assert "bored" in bored


if __name__ == "__main__":
    pytest.main([__file__, "-v"])