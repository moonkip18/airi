"""Tests for the systems added/extended in the Airi integration pass:

working memory, targeting, attention, topic interest, boredom, WAIT action,
memory maintenance (decay/reinforcement/consolidation), social graph,
response validation / anti-repetition, and delayed actions.
"""
import pytest
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.database import (
    Base, User, Conversation, Message, Memory, MemoryType, Relationship,
    ConversationStateModel, ConversationState, EmotionalState,
)
from app.memory.store import (
    get_or_create_user, get_or_create_conversation, MemoryStore,
)
from app.memory.maintenance import MemoryMaintenance
from app.state.manager import ConversationStateManager, EmotionalStateManager, RelationshipManager
from app.state.working_memory import WorkingMemoryManager
from app.state.topic_interest import TopicInterestManager
from app.state.social_graph import SocialGraphManager
from app.social.decision import SocialDecisionEngine, ParticipationFactors
from app.social.attention import AttentionScorer, attention_scorer
from app.social.delayed import DelayedActionScheduler
from app.social.validation import ResponseValidator, AntiRepetitionTracker
from app.analysis.analyzer import analyze_message
from app.analysis.targeting import classify_target, MessageTarget


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session
    await engine.dispose()


async def _conv(session):
    conv = await get_or_create_conversation(session, -100123, "group", "T")
    return conv


# ---------------------------------------------------------------------------
# Working memory
# ---------------------------------------------------------------------------
class TestWorkingMemory:
    async def test_message_and_active_participants(self, session):
        u1 = await get_or_create_user(session, 111, "alex")
        u2 = await get_or_create_user(session, 222, "sarah")
        conv = await _conv(session)
        wm = WorkingMemoryManager(session)

        await wm.add_message(conv.id, u1.id, "alex", "ok so valorant is actually broken", reply_target_user_id=None)
        await wm.add_message(conv.id, u2.id, "sarah", "minecraft better fr")
        active = await wm.get_active_participants(conv.id)
        assert set(active) == {u1.id, u2.id}

    async def test_expiration(self, session):
        u1 = await get_or_create_user(session, 111, "alex")
        conv = await _conv(session)
        wm = WorkingMemoryManager(session)

        # Force the entry to already be expired.
        entry = await wm.add_message(conv.id, u1.id, "alex", "expired soon")
        entry.expires_at = datetime.utcnow() - timedelta(minutes=1)
        await session.flush()

        msgs = await wm.get_messages(conv.id)
        assert len(msgs) == 0

        fresh = await wm.add_message(conv.id, u1.id, "alex", "fresh")
        msgs = await wm.get_messages(conv.id, limit=10)
        assert any(m.payload.get("text") == "fresh" for m in msgs)

    async def test_reply_target_and_involvement(self, session):
        u1 = await get_or_create_user(session, 111, "alex")
        u2 = await get_or_create_user(session, 222, "sarah")
        conv = await _conv(session)
        wm = WorkingMemoryManager(session)

        await wm.add_reply_target(conv.id, u1.id, u2.id)
        await wm.add_airi_action(conv.id, "reply", {"text": "lol"})
        await wm.set_involvement(conv.id, "participating")

        targets = await wm.get_reply_targets(conv.id)
        assert targets and targets[0]["from_user_id"] == u1.id and targets[0]["to_user_id"] == u2.id
        actions = await wm.get_airi_actions(conv.id)
        assert any(a["action"] == "reply" for a in actions)
        assert await wm.get_involvement(conv.id) == "participating"


# ---------------------------------------------------------------------------
# Targeting
# ---------------------------------------------------------------------------
class TestTargeting:
    def test_direct_mention(self):
        assert classify_target(is_direct_mention=True) == MessageTarget.DIRECTLY_TO_AIRI

    def test_general_group(self):
        assert classify_target(is_direct_mention=False, message_type="text") == MessageTarget.GENERAL_GROUP_MESSAGE

    def test_reply_to_another(self):
        t = classify_target(is_direct_mention=False, reply_target_user_id=5, speaker_user_id=3)
        assert t == MessageTarget.REPLY_TO_ANOTHER_USER

    def test_a_message_for_someone_else_stays_quiet(self):
        t = classify_target(is_direct_mention=False, mentioned_user_ids=[9], speaker_user_id=3)
        assert t == MessageTarget.TO_ANOTHER_USER


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class TestAttention:
    def test_high_interest_vs_boring_topic(self):
        scorer = AttentionScorer()
        minecraft = analyze_message("minecraft update is actually so good")
        boring = analyze_message("the mitochondria is the powerhouse of the cell")
        a = scorer.score(minecraft, MessageTarget.GENERAL_GROUP_MESSAGE,
                         {"minecraft": 0.9}, None, {}, current_topic="minecraft")
        b = scorer.score(boring, MessageTarget.GENERAL_GROUP_MESSAGE,
                         {"minecraft": 0.9}, None, {}, current_topic="science")
        # Explicit, deterministic check on the topic-interest sub-score.
        assert a.topic_interest > b.topic_interest
        assert a.attention_score > b.attention_score

    def test_direct_mention_boosts(self):
        scorer = AttentionScorer()
        msg = analyze_message("hey")
        direct = scorer.score(msg, MessageTarget.DIRECTLY_TO_AIRI, {}, None, {})
        general = scorer.score(msg, MessageTarget.GENERAL_GROUP_MESSAGE, {}, None, {})
        assert direct.attention_score > general.attention_score

    def test_boredom_reduces_attention(self):
        scorer = AttentionScorer()
        msg = analyze_message("valorant")
        engaged = scorer.score(msg, MessageTarget.GENERAL_GROUP_MESSAGE, {}, None, {"boredom": 0.1})
        bored = scorer.score(msg, MessageTarget.GENERAL_GROUP_MESSAGE, {}, None, {"boredom": 0.9})
        assert bored.attention_score < engaged.attention_score


# ---------------------------------------------------------------------------
# Topic interest evolution
# ---------------------------------------------------------------------------
class TestTopicInterest:
    async def test_gradual_change(self, session):
        mgr = TopicInterestManager(session)
        await mgr.touch("minecraft", engagement=0.8, positive=True)
        await mgr.touch("minecraft", engagement=0.8, positive=True)
        v1 = await mgr.get_interest("minecraft")
        assert v1 > 0.5

        # Repeated boring exposure drives it back down.
        before = await mgr.get_interest("science")
        await mgr.touch("science", engagement=0.5, positive=False)
        after = await mgr.get_interest("science")
        assert after < before

    async def test_interest_survives(self, session):
        mgr = TopicInterestManager(session)
        await mgr.touch("valorant", engagement=0.9, positive=True)
        vals = await mgr.get_all()
        assert "valorant" in vals


# ---------------------------------------------------------------------------
# Social graph
# ---------------------------------------------------------------------------
class TestSocialGraph:
    async def test_observe_and_query(self, session):
        u1 = await get_or_create_user(session, 111, "alex")
        u2 = await get_or_create_user(session, 222, "sarah")
        u3 = await get_or_create_user(session, 333, "bob")
        mgr = SocialGraphManager(session)

        await mgr.observe_interaction(u1.id, u2.id, warmth=0.5, intensity=1.0)
        await mgr.observe_interaction(u1.id, u2.id, warmth=0.5, intensity=1.0)
        closeness = await mgr.closeness_between(u1.id, u2.id)
        assert closeness > 0.1

        summaries = await mgr.get_edge_summaries([u1.id])
        assert u1.id in summaries
        assert len(summaries[u1.id]["bonds"]) >= 1

        # Non-interacting pair stays near zero.
        assert await mgr.closeness_between(u1.id, u3.id) == 0.0

    async def test_bidirectional_ordering(self, session):
        u1 = await get_or_create_user(session, 1, "a")
        u2 = await get_or_create_user(session, 2, "b")
        mgr = SocialGraphManager(session)
        await mgr.observe_interaction(u1.id, u2.id, warmth=0.4)
        e = await mgr.get_edge(u1.id, u2.id)
        assert e is not None
        e2 = await mgr.get_edge(u2.id, u1.id)
        assert e2 is not None and e2.id == e.id


# ---------------------------------------------------------------------------
# Boredom (stateful) + state decay
# ---------------------------------------------------------------------------
class TestBoredomState:
    async def test_boredom_is_stateful_and_decays(self, session):
        mgr = EmotionalStateManager(session)
        # Boredom builds gradually over repeated boring exposure.
        for _ in range(10):
            await mgr.update_from_event("boring", intensity=1.0)
        state = await mgr.get_state()
        assert state["boredom"] > 0.25  # climbed well above the 0.2 baseline

        await mgr.apply_decay()
        state2 = await mgr.get_state()
        assert state2["boredom"] < state["boredom"]

    async def test_interesting_drops_boredom(self, session):
        mgr = EmotionalStateManager(session)
        await mgr.update_from_event("boring", intensity=1.0)
        before = (await mgr.get_state())["boredom"]
        await mgr.update_from_event("interesting_topic", intensity=1.0)
        after = (await mgr.get_state())["boredom"]
        assert after < before


# ---------------------------------------------------------------------------
# Social decision engine: WAIT vs IGNORE
# ---------------------------------------------------------------------------
class TestDecisionEngine:
    def test_wait_vs_ignore(self):
        engine = SocialDecisionEngine()
        # Low attention + low score -> ignore
        assert engine.decide_passive_action(score=0.1, attention=0.1, boredom=0.2) == "ignore"
        # Moderate attention but just below action threshold -> wait
        assert engine.decide_passive_action(score=0.35, attention=0.5, boredom=0.2) == "wait"
        # High boredom driving away
        assert engine.decide_passive_action(score=0.1, attention=0.1, boredom=0.9, want_to_leave=True) == "leave"


# ---------------------------------------------------------------------------
# Memory maintenance: reinforcement, decay, consolidation, correction
# ---------------------------------------------------------------------------
class TestMemoryMaintenance:
    async def test_reinforcement(self, session):
        u = await get_or_create_user(session, 123, "alex")
        store = MemoryStore(session)
        mem = await store.create_memory(
            memory_type=MemoryType.PREFERENCE,
            content="Alex strongly prefers Minecraft",
            user_id=u.id,
            people_involved=[123],
            topics=["minecraft"],
            importance=0.7, confidence=0.8,
        )
        maintenance = MemoryMaintenance(session)
        # A close re-statement should match and reinforce, not duplicate.
        match = await maintenance.find_match("Alex strongly prefers Minecraft", user_id=u.id)
        assert match is not None and match.id == mem.id
        await maintenance.reinforce(match, "Alex strongly prefers Minecraft")
        assert match.reinforcement >= 1
        assert match.confidence > 0.8

    async def test_decay_lowers_old_low_importance(self, session):
        u = await get_or_create_user(session, 123, "alex")
        store = MemoryStore(session)
        old = await store.create_memory(
            memory_type=MemoryType.EPISODIC,
            content="some trivial old thing",
            user_id=u.id,
            importance=0.4, confidence=0.5,
        )
        old.created_at = datetime.utcnow() - timedelta(days=10)
        relevant = await store.create_memory(
            memory_type=MemoryType.PREFERENCE,
            content="Alex likes minecraft a lot",
            user_id=u.id,
            importance=0.8, confidence=0.9, reinforcement=5,
        )
        relevant.created_at = datetime.utcnow() - timedelta(days=10)
        await session.flush()

        maintenance = MemoryMaintenance(session)
        await maintenance.decay(factor=0.9)

        triv = (await store.get_memory(old.id))
        imp = (await store.get_memory(relevant.id))
        assert triv.importance < 0.4
        # Reinforced important memory is protected.
        assert imp.importance == 0.8

    async def test_consolidation(self, session):
        u = await get_or_create_user(session, 123, "alex")
        conv = await _conv(session)
        store = MemoryStore(session)
        for i in range(4):
            await store.create_memory(
                memory_type=MemoryType.EPISODIC,
                content=f"Alex played Minecraft episode {i}",
                user_id=u.id, conversation_id=conv.id,
                people_involved=[123],
                topics=["minecraft"],
                importance=0.6, confidence=0.8,
            )
        maintenance = MemoryMaintenance(session)
        created = await maintenance.consolidate()
        assert created >= 1
        # Original episodes preserved.
        from sqlalchemy import select, func
        from app.database import Memory as _M
        stmt = select(func.count(_M.id)).where(_M.memory_type == MemoryType.EPISODIC)
        assert (await session.execute(stmt)).scalar() == 4

    async def test_correction_supersedes(self, session):
        u = await get_or_create_user(session, 123, "alex")
        store = MemoryStore(session)
        old = await store.create_memory(
            memory_type=MemoryType.SEMANTIC,
            content="Alex hates physics",
            user_id=u.id, people_involved=[123],
            topics=["physics"], importance=0.7,
        )
        # New contradictory info supersedes the old belief. (find_contradictory
        # is embedding-based, so use a strongly overlapping statement to make
        # the supersede mechanism deterministic across embedding backends.)
        contradictory = await store.find_contradictory_memories("alex hates physics", user_id=u.id, similarity_threshold=0.7)
        assert any(m.id == old.id for m in contradictory)
        new = await store.create_memory(
            memory_type=MemoryType.SEMANTIC,
            content="Alex loves physics now",
            user_id=u.id, people_involved=[123],
            topics=["physics"], importance=0.75,
        )
        await store.supersede_memory(old, new)
        assert old.superseded_by == new.id
        # Superseded memories are excluded from normal retrieval.
        from sqlalchemy import select
        from app.database import Memory as _M
        res = await session.execute(select(_M).where(_M.id == old.id, _M.superseded_by.is_(None)))
        assert res.scalar_one_or_none() is None


# ---------------------------------------------------------------------------
# Relationships: negative interaction + apology/forgiveness
# ---------------------------------------------------------------------------
class TestRelationshipDynamics:
    async def test_disappointment_then_apology(self, session):
        u = await get_or_create_user(session, 123, "alex")
        mgr = RelationshipManager(session)
        # Build up fondness.
        await mgr.update_from_interaction(u.id, "positive", 2.0)
        happy = await mgr.get_relationship_summary(u.id)

        await mgr.update_from_interaction(u.id, "disappointment", 1.0)
        disappointed = await mgr.get_relationship_summary(u.id)
        assert disappointed["affection"] < happy["affection"]
        assert disappointed["annoyance"] > happy["annoyance"]

        # Forgiveness is graded: an apology restores a lot without echoing.
        await mgr.update_from_interaction(u.id, "apology", 1.0)
        forgiven = await mgr.get_relationship_summary(u.id)
        assert forgiven["annoyance"] < disappointed["annoyance"]
        assert forgiven["affection"] > disappointed["affection"]


# ---------------------------------------------------------------------------
# Response validation + anti-repetition
# ---------------------------------------------------------------------------
class TestResponseValidation:
    async def test_blocks_ai_filler(self, session):
        v = ResponseValidator(session)
        res = await v.validate("As an AI, I'm here to help you in any way I can!", conversation_id=1)
        assert res["ok"] is False
        assert await v.should_skip(res) is True

    async def test_blocks_long_and_repeats(self, session):
        v = ResponseValidator(session)
        tracker = AntiRepetitionTracker(session)
        long_resp = "x" * 1000
        res = await v.validate(long_resp, conversation_id=1)
        assert any(i.startswith("too_long") for i in res["issues"])
        # Over-length casual replies are trimmed, not just flagged.
        assert len(res["response"]) <= 400

        await tracker.record(1, "response_pattern", "wait what lol")
        dup = await v.validate("wait what lol", conversation_id=1)
        assert "repetitive" in dup["issues"]

    async def test_allows_natural_response(self, session):
        v = ResponseValidator(session)
        res = await v.validate("oh my god you're so annoying lol", current_topic="valorant", conversation_id=1)
        assert res["ok"] is True


# ---------------------------------------------------------------------------
# Delayed actions
# ---------------------------------------------------------------------------
class TestDelayedActions:
    async def test_schedule_and_due(self, session):
        sched = DelayedActionScheduler(session)
        a = await sched.schedule_in(None, "return_to_conversation", delay_seconds=-1.0, reason="test")
        due = await sched.get_due()
        assert any(x.id == a.id for x in due)
        await sched.mark_executed(a)
        due = await sched.get_due()
        assert all(x.id != a.id for x in due)

    async def test_count_pending(self, session):
        sched = DelayedActionScheduler(session)
        await sched.schedule_in(1, "react", delay_seconds=5.0)
        await sched.schedule_in(2, "follow_up", delay_seconds=5.0)
        assert await sched.count_pending() == 2
        assert await sched.count_pending(1) == 1


# ---------------------------------------------------------------------------
# Persistence across "restart" (new session on same DB)
# ---------------------------------------------------------------------------
class TestPersistence:
    async def test_state_and_interests_survive_restart(self):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        try:
            async with async_session() as s1:
                await s1.commit()
                u = await get_or_create_user(s1, 42, "alex")
                mgr = TopicInterestManager(s1)
                await mgr.touch("minecraft", engagement=1.0, positive=True)
                emo = EmotionalStateManager(s1)
                await emo.update_from_event("funny", 1.0)
                await s1.commit()

            # New session == simulated restart.
            async with async_session() as s2:
                interest = await TopicInterestManager(s2).get_interest("minecraft")
                state = await EmotionalStateManager(s2).get_state()
                assert interest > 0.5
                assert state["amusement"] > 0
        finally:
            await engine.dispose()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])