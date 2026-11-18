"""Tests for realism round: ping fatigue (#7), memory callbacks (#6),
autobiographical diary (#3), style audit (#10), late typo fix (#5).
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from app.database import (
    Memory, MemoryType, Relationship, SelfModel, User,
)
from app.state.manager import RelationshipManager
from app.agent.timeutil import _age_label
from app.config import settings


# ---------------------------------------------------------------------------
# #7 ping fatigue
# ---------------------------------------------------------------------------
class TestPingFatigue:
    async def test_rapid_messages_stack(self, session):
        mgr = RelationshipManager(session)
        for _ in range(4):
            streak, crossed = await mgr.register_message_pulse(1)
        # First message sets no fatigue; 3 further rapid x 0.34 = 1.02 >= 1.0
        assert streak >= 1.0
        assert crossed is True

    async def test_crossing_fires_once(self, session):
        mgr = RelationshipManager(session)
        results = []
        for _ in range(5):
            streak, crossed = await mgr.register_message_pulse(1)
            results.append(crossed)
        assert results.count(True) == 1

    async def test_time_gap_decays(self, session):
        mgr = RelationshipManager(session)
        await mgr.register_message_pulse(1)
        await mgr.register_message_pulse(1)
        rel = await mgr.get_relationship(1)
        # Simulate 3h of quiet: decay 0.6/h wipes the streak.
        rel.last_message_at = datetime.utcnow() - timedelta(hours=3)
        streak, crossed = await mgr.register_message_pulse(1)
        assert streak == 0.0
        assert crossed is False

    async def test_cap_at_max(self, session):
        mgr = RelationshipManager(session)
        for _ in range(30):
            streak, _ = await mgr.register_message_pulse(1)
        assert streak <= settings.irritation_max

    async def test_summary_hides_stale_streak(self, session):
        mgr = RelationshipManager(session)
        await mgr.register_message_pulse(1)
        await mgr.register_message_pulse(1)
        await mgr.register_message_pulse(1)
        rel = await mgr.get_relationship(1)
        # Stale beyond 2x window -> summary reports 0 irritation.
        rel.last_message_at = datetime.utcnow() - timedelta(hours=2)
        summary = await mgr.get_relationship_summary(1)
        assert summary["irritation_streak"] == 0.0

        # Fresh -> summary carries it.
        rel.last_message_at = datetime.utcnow()
        summary = await mgr.get_relationship_summary(1)
        assert summary["irritation_streak"] > 0.0


# ---------------------------------------------------------------------------
# #2/_age_label
# ---------------------------------------------------------------------------
class TestAgeLabel:
    def test_buckets(self):
        now = datetime.utcnow()
        assert _age_label(now) == "just now"
        assert _age_label(now - timedelta(hours=3)) == "3h ago"
        assert _age_label(now - timedelta(days=3)) == "3d ago"
        assert _age_label(now - timedelta(days=14)) == "2w ago"
        assert _age_label(now - timedelta(days=90)) == "3mo ago"
        assert _age_label(now - timedelta(days=800)) == "2y ago"
        assert _age_label(None) == "a while ago"


# ---------------------------------------------------------------------------
# #6 memory callbacks
# ---------------------------------------------------------------------------
def _mk_retrieved(memory, relevance=0.8):
    rm = MagicMock()
    rm.memory = memory
    rm.relevance_score = relevance
    return rm


class TestMemoryCallbacks:
    def _mk_memory(self, *, mtype=MemoryType.EPISODIC, age_days=10,
                   content="went to a rock concert with Amir",
                   last_retrieved=None):
        m = MagicMock(spec=Memory)
        m.memory_type = mtype
        m.created_at = datetime.utcnow() - timedelta(days=age_days)
        m.content = content
        m.last_retrieved = last_retrieved
        m.id = 42
        return m

    async def test_picks_old_relevant_episodic(self, session):
        from app.agent.core import Agent
        agent = Agent.__new__(Agent)  # no init; only test the picker
        agent.session = session
        agent.anti_repetition = MagicMock()
        agent.anti_repetition.record = AsyncMock()
        mem = self._mk_memory()
        cb = await Agent._pick_callback(
            agent, [_mk_retrieved(mem)], conversation_id=1,
        )
        assert cb is not None
        assert cb["age_label"] == "1w ago"
        assert "rock concert" in cb["text"]

    async def test_rejects_fresh_memories(self, session):
        from app.agent.core import Agent
        agent = Agent.__new__(Agent)
        agent.session = session
        agent.anti_repetition = MagicMock()
        agent.anti_repetition.record = AsyncMock()
        mem = self._mk_memory(age_days=0)  # today: not a story yet
        cb = await Agent._pick_callback(
            agent, [_mk_retrieved(mem)], conversation_id=1,
        )
        assert cb is None

    async def test_rejects_low_relevance(self, session):
        from app.agent.core import Agent
        agent = Agent.__new__(Agent)
        agent.session = session
        agent.anti_repetition = MagicMock()
        agent.anti_repetition.record = AsyncMock()
        mem = self._mk_memory()
        cb = await Agent._pick_callback(
            agent, [_mk_retrieved(mem, relevance=0.05)], conversation_id=1,
        )
        assert cb is None

    async def test_refractory_persists(self, session):
        from app.agent.core import Agent
        agent = Agent.__new__(Agent)
        agent.session = session
        agent.anti_repetition = MagicMock()
        agent.anti_repetition.record = AsyncMock()
        mem = self._mk_memory()
        cb1 = await Agent._pick_callback(
            agent, [_mk_retrieved(mem)], conversation_id=1,
        )
        assert cb1 is not None
        # Immediately after: refractory window blocks a repeat.
        mem.last_retrieved = datetime.utcnow()
        cb2 = await Agent._pick_callback(
            agent, [_mk_retrieved(mem)], conversation_id=1,
        )
        assert cb2 is None


# ---------------------------------------------------------------------------
# #3 autobiographical diary
# ---------------------------------------------------------------------------
class TestDiary:
    async def test_collects_bot_owned_episodic(self, session):
        from app.agent.core import Agent
        bot_user = User(telegram_id=777000000, username="Airi")
        session.add(bot_user)
        await session.flush()
        session.add(Memory(
            memory_type=MemoryType.EPISODIC,
            content="stayed up grinding p5 all night",
            user_id=bot_user.id,
            importance=0.6,
            source="extraction",
        ))
        session.add(Memory(  # someone else's — excluded
            memory_type=MemoryType.EPISODIC,
            content="amir passed his exam",
            user_id=None,
            importance=0.6,
            source="extraction",
        ))
        await session.flush()

        agent = Agent.__new__(Agent)
        agent.session = session
        diary = await Agent._collect_recent_diary(agent)
        assert len(diary) == 1
        assert "p5" in diary[0]["text"]


# ---------------------------------------------------------------------------
# #10 style audit
# ---------------------------------------------------------------------------
class TestStyleAudit:
    def _mk_background(self, texts):
        from app.agent.background import BackgroundTasks
        bg = BackgroundTasks.__new__(BackgroundTasks)
        bg._chat_filter = lambda: None
        session = MagicMock()
        session.execute = AsyncMock(
            return_value=[(t,) for t in texts]
        )
        # The loop code does `.execute(stmt.limit(500)).all()`
        result = MagicMock()
        result.all = MagicMock(return_value=[(t,) for t in texts])
        session.execute = AsyncMock(return_value=result)
        return bg, session

    async def test_detects_opener_tic(self):
        bg, session = self._mk_background(
            ["wait what", "wait why", "wait how", "no idea"] * 4
        )  # 16 msgs, 75% open with 'wait'
        alarms = await bg._compute_style_alarms(session)
        assert alarms is not None
        assert any("wait" in a for a in alarms["alarms"])

    async def test_detects_question_rate(self):
        bg, session = self._mk_background(["really?", "why tho?", "huh?", "lol"] * 5)
        alarms = await bg._compute_style_alarms(session)
        assert alarms is not None
        assert any("questions" in a for a in alarms["alarms"])

    async def test_quiet_week_skipped(self):
        bg, session = self._mk_background(["lol", "yeah"])  # 2 messages < 15
        alarms = await bg._compute_style_alarms(session)
        assert alarms is None


# ---------------------------------------------------------------------------
# #5 late typo fix
# ---------------------------------------------------------------------------
class TestLateTypoFixConfig:
    def test_settings_exist(self):
        assert 0 <= settings.late_typo_fix_chance <= 1
        assert settings.late_typo_fix_min_seconds > 10
        assert settings.late_typo_fix_max_seconds > settings.late_typo_fix_min_seconds
