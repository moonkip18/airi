import asyncio
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from app.database.session import db_write_lock
from app.config import settings
from app.database import (
    Conversation, ConversationState, EmotionalState,
    get_session
)
from app.agent import Agent
from app.state import EmotionalStateManager

logger = structlog.get_logger()


class BackgroundTasks:
    def __init__(self, bot):
        self.bot = bot
        self._running = False
        self._tasks: list[asyncio.Task] = []
        # Cost control: memory extraction / learning are LLM calls, so they
        # only run when new messages arrived since the last run — not blindly
        # on every interval tick. The B.AI token budget is generous (30M), so
        # the gates are loose (quality-first); they exist to skip SILENT
        # chats, not to ration normal activity.
        self._last_extraction_msg: dict[int, int] = {}
        self._last_learning_msg: dict[int, int] = {}

    async def start(self):
        self._running = True

        self._tasks.append(asyncio.create_task(self._emotional_decay_loop()))
        self._tasks.append(asyncio.create_task(self._memory_extraction_loop()))
        self._tasks.append(asyncio.create_task(self._learning_loop()))
        self._tasks.append(asyncio.create_task(self._conversation_state_loop()))
        self._tasks.append(asyncio.create_task(self._autonomous_scan_loop()))
        self._tasks.append(asyncio.create_task(self._memory_maintenance_loop()))
        self._tasks.append(asyncio.create_task(self._delayed_action_loop()))
        self._tasks.append(asyncio.create_task(self._conversation_summary_loop()))
        self._tasks.append(asyncio.create_task(self._style_audit_loop()))

        logger.info("background_tasks_started")

    async def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("background_tasks_stopped")

    async def _emotional_decay_loop(self):
        while self._running:
            try:
                await asyncio.sleep(60)
                async with db_write_lock, get_session() as session:
                    manager = EmotionalStateManager(session)
                    await manager.apply_decay()
                    # Body clock: energy/social pull toward the local
                    # time-of-day baseline.
                    await manager.apply_circadian()
                    await session.commit()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("emotional_decay_error", error=str(e))

    async def _new_messages_since(self, session, conversation_id: int, last_id: int) -> int:
        from sqlalchemy import func
        from app.database import Message as _Message
        return await session.scalar(
            select(func.count(_Message.id)).where(
                _Message.conversation_id == conversation_id,
                _Message.id > last_id,
            )
        ) or 0

    def _chat_filter(self):
        """Access control for background loops: when ALLOWED_CHAT_IDS is set,
        conversations from any other chat are excluded from ALL background
        work (no LLM cost for chats the owner didn't choose)."""
        from app.config import get_allowed_chat_ids
        allowed = get_allowed_chat_ids()
        if allowed is None:
            return None
        return Conversation.telegram_chat_id.in_(allowed)

    async def _latest_message_id(self, session, conversation_id: int) -> int:
        from sqlalchemy import func
        from app.database import Message as _Message
        return await session.scalar(
            select(func.max(_Message.id)).where(
                _Message.conversation_id == conversation_id,
            )
        ) or 0

    async def _memory_extraction_loop(self):
        while self._running:
            try:
                await asyncio.sleep(settings.memory_extraction_interval)
                # NO db_write_lock around the session: this loop used to hold
                # the global write lock across the LLM call (starving every
                # message handler for 30s+) and then RE-ACQUIRE it for the
                # commit — a guaranteed self-deadlock on the non-reentrant
                # asyncio.Lock that froze the whole bot. Reads are safe in WAL
                # mode without the lock; only the final flush takes it.
                async with get_session() as session:
                    stmt = select(Conversation).options(selectinload(Conversation.state)).where(
                        Conversation.updated_at > datetime.utcnow() - timedelta(hours=1)
                    )
                    flt = self._chat_filter()
                    if flt is not None:
                        stmt = stmt.where(flt)
                    result = await session.execute(stmt)
                    conversations = result.scalars().all()

                    for conv in conversations:
                        if conv.state and conv.state.state in [
                            ConversationState.PARTICIPATING,
                            ConversationState.WINDING_DOWN,
                        ]:
                            # Skip the LLM call only when the conversation is
                            # silent (few new messages since last run).
                            last_id = self._last_extraction_msg.get(conv.id, 0)
                            if last_id and await self._new_messages_since(session, conv.id, last_id) < 4:
                                continue
                            agent = self.bot._get_agent(session)
                            # LLM call runs WITHOUT db_write_lock (it can take
                            # 30s+; holding the global lock that long starves
                            # every other writer and cascades into
                            # "database is locked"). Writes are serialized by
                            # the commit below instead.
                            await agent.run_memory_extraction(conv.id)
                            self._last_extraction_msg[conv.id] = await self._latest_message_id(session, conv.id)
                            # Serialize ONLY the flush — brief acquisition of
                            # a now-free lock, not a self-deadlock.
                            async with db_write_lock:
                                await session.commit()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("memory_extraction_loop_error", error=str(e))

    async def _learning_loop(self):
        """Periodic learning: what happened, what did Airi learn?

        One batched evaluation per active conversation window; also slowly
        decays topic engagement and relationship jealousy.
        """
        while self._running:
            try:
                await asyncio.sleep(settings.learning_loop_interval)
                # Same fix as the extraction loop: no db_write_lock around the
                # session — the LLM call must not hold the global lock, and
                # the tail block below used to RE-ACQUIRE it in the same task
                # (unconditionally, every tick) = instant self-deadlock.
                async with get_session() as session:
                    stmt = select(Conversation).options(selectinload(Conversation.state)).where(
                        Conversation.updated_at > datetime.utcnow() - timedelta(hours=2)
                    )
                    flt = self._chat_filter()
                    if flt is not None:
                        stmt = stmt.where(flt)
                    result = await session.execute(stmt)
                    conversations = result.scalars().all()

                    for conv in conversations:
                        if not conv.state:
                            continue
                        if conv.state.state in [
                            ConversationState.PARTICIPATING,
                            ConversationState.WINDING_DOWN,
                            ConversationState.OBSERVING,
                        ]:
                            # Same rule as extraction: skip silent windows.
                            last_id = self._last_learning_msg.get(conv.id, 0)
                            if last_id and await self._new_messages_since(session, conv.id, last_id) < 3:
                                continue
                            agent = self.bot._get_agent(session)
                            # LLM call outside db_write_lock (see extraction loop).
                            await agent.run_learning_loop(conv.id)
                            self._last_learning_msg[conv.id] = await self._latest_message_id(session, conv.id)

                    # Slow decay of obsessions/engagement + jealousy. These
                    # WRITES (plus the learning writes above, still uncommitted
                    # in this session) go in under the lock, then commit —
                    # the write transaction stays short. (Acquiring here is
                    # safe: the outer session no longer holds the lock.)
                    async with db_write_lock:
                        from app.agent.learning import SelfModelManager
                        await SelfModelManager(session).decay_engagement(factor=0.97)
                        from app.state import RelationshipManager
                        rel_mgr = RelationshipManager(session)
                        await rel_mgr.decay_jealousy()
                        # Deterministic evidence accumulation: repeated interaction
                        # deepens bonds even when the LLM learning pass says nothing.
                        await rel_mgr.strengthen_from_evidence()

                        await session.commit()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("learning_loop_error", error=str(e))

    async def _conversation_state_loop(self):
        while self._running:
            try:
                await asyncio.sleep(300)
                async with db_write_lock, get_session() as session:
                    stmt = select(Conversation).options(selectinload(Conversation.state))
                    flt = self._chat_filter()
                    if flt is not None:
                        stmt = stmt.where(flt)
                    result = await session.execute(stmt)
                    conversations = result.scalars().all()

                    for conv in conversations:
                        if not conv.state:
                            continue

                        if conv.state.state == ConversationState.PARTICIPATING:
                            from app.state import ConversationStateManager
                            manager = ConversationStateManager(session)
                            winding_down = await manager.detect_winding_down(conv.id)
                            if winding_down > 0.8:
                                await manager.update_state(
                                    conv.id,
                                    new_state=ConversationState.WINDING_DOWN,
                                    winding_down_score=winding_down,
                                )
                                await session.commit()

                        elif conv.state.state == ConversationState.WINDING_DOWN:
                            # Fully drift away after the wind-down period.
                            updated = conv.state.updated_at or conv.updated_at
                            if updated and datetime.utcnow() - updated > timedelta(minutes=10):
                                conv.state.state = ConversationState.AWAY
                                conv.state.is_participating = False
                                conv.state.is_observing = False
                                await session.commit()
                                logger.info(
                                    "conversation_fully_away",
                                    conversation_id=conv.id,
                                )

                        elif conv.state.state in (ConversationState.OBSERVING, ConversationState.INTERESTED):
                            # A bored or long-idle observer eventually drifts
                            # away — otherwise she could linger observing
                            # forever and the AWAY->OBSERVING cycle never runs.
                            from app.state import ConversationStateManager as _CSM, EmotionalStateManager as _ESM
                            emo = await EmotionalStateManager(session).get_state()
                            moved = await _CSM(session).maybe_observing_away(
                                conv.state, emo["boredom"]
                            )
                            if moved:
                                await session.commit()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("conversation_state_loop_error", error=str(e))

    async def _autonomous_scan_loop(self):
        while self._running:
            try:
                await asyncio.sleep(settings.autonomous_check_interval * 2)
                if not settings.autonomous_behavior_enabled:
                    continue

                async with db_write_lock, get_session() as session:
                    stmt = select(Conversation).options(selectinload(Conversation.state)).where(
                        Conversation.chat_type.in_(["group", "supergroup"])
                    )
                    flt = self._chat_filter()
                    if flt is not None:
                        stmt = stmt.where(flt)
                    result = await session.execute(stmt)
                    conversations = result.scalars().all()

                    for conv in conversations:
                        if not conv.state:
                            continue

                        if conv.state.state in [ConversationState.AWAY, ConversationState.OBSERVING]:
                            await self.bot.start_autonomous_loop(conv.id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("autonomous_scan_loop_error", error=str(e))

    async def _memory_maintenance_loop(self):
        """Periodic memory decay + consolidation + dead-memory pruning."""
        while self._running:
            try:
                await asyncio.sleep(settings.memory_maintenance_interval)
                async with db_write_lock, get_session() as session:
                    from app.memory.maintenance import MemoryMaintenance
                    maintenance = MemoryMaintenance(session)
                    # One-time migration after an embedding-backend switch
                    # (e.g. sentence-transformers just became available):
                    # old fallback vectors are the wrong dimensionality and
                    # invisible to retrieval until re-encoded.
                    await maintenance.reembed_stale_backend()
                    await maintenance.decay()
                    # Consolidation is heavier; run it less often.
                    if await self._consolidation_due(session):
                        await maintenance.consolidate()
                    await maintenance.prune_dead()
                    await session.commit()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("memory_maintenance_loop_error", error=str(e))

    async def _consolidation_due(self, session) -> bool:
        """Consolidation runs every consolidation_interval seconds."""
        from datetime import datetime, timedelta
        from app.config import settings as _settings
        key = "last_memory_consolidation"
        now = datetime.utcnow()
        last = getattr(self, key, None)
        if last is None or (now - last).total_seconds() >= _settings.memory_consolidation_interval:
            setattr(self, key, now)
            return True
        return False

    async def _style_audit_loop(self):
        """Weekly self-style audit (#10): she measures her OWN drift.

        Deterministic stats over her recent messages (no LLM, no cost):
        overused openers, question rate, length creep. Alarms land in
        self_model.style_alarms and render in her prompt's self-development
        section, so she can correct her own habits — the same way a person
        notices they've been starting every message with 'wait' lately.
        First run is deliberately soon; afterwards it's weekly.
        """
        first = True
        while self._running:
            try:
                delay = (
                    settings.style_audit_first_delay_seconds
                    if first
                    else settings.style_audit_interval_seconds
                )
                first = False
                await asyncio.sleep(delay)

                async with get_session() as session:
                    # Reads need no lock (WAL); compute, then write briefly.
                    alarms = await self._compute_style_alarms(session)
                    if alarms is None:
                        continue
                    async with db_write_lock:
                        from app.agent.learning import SelfModelManager
                        model = await SelfModelManager(session).get_or_create()
                        model.style_alarms = alarms
                        await session.commit()
                    logger.info(
                        "style_audit_complete",
                        n_messages=alarms.get("n_messages", 0),
                        alarms=len(alarms.get("alarms", [])),
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("style_audit_loop_error", error=str(e))

    async def _compute_style_alarms(self, session) -> dict | None:
        """Measure her recent voice for drift. Pure function of the DB —
        testable without Telegram or LLM. Returns the alarms payload or
        None when there's not enough to judge."""
        from sqlalchemy import select as _select, func as _func
        from app.database import Message, Conversation
        from datetime import timedelta as _td

        cutoff = datetime.utcnow() - _td(days=settings.style_audit_window_days)
        stmt = (
            select(Message.text)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Message.is_from_bot.is_(True),
                Message.text.isnot(None),
                Message.timestamp > cutoff,
            )
        )
        flt = self._chat_filter()
        if flt is not None:
            stmt = stmt.where(flt)
        texts = [t for (t,) in (await session.execute(stmt.limit(500))).all()]

        n = len(texts)
        if n < 15:
            return None  # too quiet a week to judge style on

        from collections import Counter as _Counter
        openers = _Counter()
        question_count = 0
        total_words = 0
        for t in texts:
            words = t.strip().split()
            if words:
                openers[words[0].lower().strip(".,!?")] += 1
                total_words += len(words)
            if "?" in t:
                question_count += 1

        avg_words = total_words / n
        question_rate = question_count / n
        alarms: list[str] = []

        # Overused openers: starting >28% of messages with the same word is
        # a verbal tic (measured baseline: humans here are far more varied).
        top_word, top_count = (openers.most_common(1) or [(None, 0)])[0]
        if top_word and top_count >= 8 and top_count / n > 0.28:
            alarms.append(
                f"you've started {int(top_count / n * 100)}% of your last {n} "
                f"messages with '{top_word}' — vary your openers"
            )

        # Question rate: the #1 measured AI tell (humans: ~3%, she was: 15%).
        if question_rate > 0.15:
            alarms.append(
                f"you asked questions in {int(question_rate * 100)}% of your "
                "recent messages — most casual messages don't need one"
            )

        # Length creep: her measured failure mode.
        if avg_words > 14:
            alarms.append(
                f"your recent replies averaged {int(avg_words)} words — "
                "cut them down; most should be a few words"
            )

        return {
            "checked_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "window_days": settings.style_audit_window_days,
            "n_messages": n,
            "avg_words": round(avg_words, 1),
            "question_rate": round(question_rate, 3),
            "alarms": alarms,
        }

    async def _conversation_summary_loop(self):
        """Rolling conversation summaries = actual continuity.

        Every ~30 min, recently active conversations with enough new traffic
        get compressed into one conversation_summary memory ("we talked about
        X and Y; Z happened"). Without this, everything before the last 30
        messages simply never happened to her.
        """
        from datetime import timedelta as _td
        from sqlalchemy import select as _select
        from app.database import Memory, MemoryType, Message, User as _User
        from app.memory.store import MemoryStore

        while self._running:
            try:
                await asyncio.sleep(1800)

                async with get_session() as session:
                    stmt = _select(Conversation).where(
                        Conversation.updated_at > datetime.utcnow() - _td(hours=3)
                    )
                    flt = self._chat_filter()
                    if flt is not None:
                        stmt = stmt.where(flt)
                    conversations = (await session.execute(stmt)).scalars().all()

                    for conv in conversations:
                        last_summary = (
                            await session.execute(
                                _select(Memory)
                                .where(
                                    Memory.conversation_id == conv.id,
                                    Memory.memory_type == MemoryType.CONVERSATION_SUMMARY,
                                )
                                .order_by(Memory.created_at.desc())
                                .limit(1)
                            )
                        ).scalar_one_or_none()

                        cutoff = (
                            last_summary.created_at
                            if last_summary
                            else datetime.utcnow() - _td(hours=24)
                        )

                        recent = (
                            await session.execute(
                                _select(Message)
                                .where(
                                    Message.conversation_id == conv.id,
                                    Message.timestamp > cutoff,
                                )
                                .order_by(Message.timestamp.asc())
                                .limit(60)
                            )
                        ).scalars().all()

                        # A summary needs a real chunk of conversation.
                        if len([m for m in recent if (m.text or "").strip()]) < 8:
                            continue

                        # Commit the reads, then release the DB before the
                        # (potentially slow) LLM call.
                        await session.commit()

                        await self._summarize_conversation(session, conv.id, recent)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("conversation_summary_loop_error", error=str(e))

    async def _summarize_conversation(self, session, conversation_id: int, messages):
        """One batched LLM call: compress a message window into a durable
        conversation_summary memory (retrievable like any other memory)."""
        from datetime import datetime as _dt
        from typing import Literal as _Literal, List as _List
        from pydantic import BaseModel as _BaseModel, Field as _Field
        from sqlalchemy import select as _select
        from app.database import Memory, MemoryType, Message, User as _User
        from app.memory.store import MemoryStore
        from app.llm import llm_provider

        user_ids = {m.user_id for m in messages if m.user_id}
        user_rows = (
            await session.execute(_select(_User).where(_User.id.in_(user_ids)))
        ).scalars().all()
        id_to_name = {
            u.id: (u.username or u.first_name or f"user{u.telegram_id}")
            for u in user_rows
        }

        transcript = "\n".join(
            f"{id_to_name.get(m.user_id, 'someone')}: {m.text}"
            for m in messages
            if (m.text or "").strip()
        )

        class SummaryOutcome(_BaseModel):
            summary: str = ""
            topics: _List[str] = _Field(default_factory=list)
            about_usernames: _List[str] = _Field(default_factory=list)

        system = (
            "You compress a group-chat window into a durable memory summary "
            "for an AI persona. Capture what happened, what was discussed, "
            "and anything notable about the people involved — the way a "
            "person would briefly recall the hangout later. Respond ONLY "
            "with valid JSON matching the schema. Write in English, 1-3 "
            "sentences, casual but informative."
        )

        prompt = (
            f"Conversation window:\n{transcript[:6000]}\n\n"
            f"username -> user_id map: {id_to_name}\n\n"
            "Summarize this window as a durable memory."
        )

        try:
            outcome = await llm_provider.generate_structured(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                SummaryOutcome,
                temperature=0.3,
                max_tokens=1000,
            )
        except Exception as e:
            logger.error("conversation_summary_failed", error=str(e))
            return

        if not outcome.summary.strip():
            return

        username_to_id = {str(v).lstrip("@").lower(): k for k, v in id_to_name.items()}
        people = [
            username_to_id[str(n).lstrip("@").lower()]
            for n in outcome.about_usernames
            if str(n).lstrip("@").lower() in username_to_id
        ]

        # Write + commit atomically under the global write lock — the same
        # discipline as every other writer (an uncommitted write tx outside
        # the lock holds SQLite's RESERVED lock and blocks all other writers).
        from app.database.session import db_write_lock as _dbwl
        async with _dbwl:
            store = MemoryStore(session)
            await store.create_memory(
                memory_type=MemoryType.CONVERSATION_SUMMARY,
                content=outcome.summary.strip()[:800],
                conversation_id=conversation_id,
                people_involved=people,
                topics=[t[:60] for t in outcome.topics[:6]],
                importance=0.55,
                confidence=0.85,
                source="summary",
            )
            await session.commit()
        logger.info(
            "conversation_summarized",
            conversation_id=conversation_id,
            messages=len(messages),
            summary_preview=outcome.summary[:80],
        )

    async def _delayed_action_loop(self):
        """Process scheduled delayed actions without blocking the event loop.

        The write transaction must NEVER span a Telegram network call: with a
        slow/proxied connection a send can hang for tens of seconds, and an
        open write tx that long makes every other writer hit
        "database is locked". So: mark executed + commit under the lock,
        then do the (possibly slow) execution with the lock released.
        """
        while self._running:
            try:
                await asyncio.sleep(settings.delayed_action_interval)
                async with get_session() as session:
                    from app.social.delayed import DelayedActionScheduler
                    scheduler = DelayedActionScheduler(session)
                    due = await scheduler.get_due()
                    for action in due:
                        async with db_write_lock:
                            await scheduler.mark_executed(action)
                            await session.commit()
                        await self._execute_delayed(session, action)
                        async with db_write_lock:
                            await session.commit()
                    async with db_write_lock:
                        await scheduler.prune()
                        await session.commit()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("delayed_action_loop_error", error=str(e))

    async def _execute_delayed(self, session, action):
        from app.database import Conversation as _Conversation, Message as _Message, User as _User
        from sqlalchemy import select as _select

        atype = action.action_type
        conv_id = action.conversation_id

        if atype == "return_to_conversation":
            # No public message — just make Airi alert again so the autonomous
            # loop can pick her up if something interesting happens.
            from app.state import ConversationStateManager
            from app.database import ConversationState as _CS
            mgr = ConversationStateManager(session)
            stmt = _select(_Conversation).options(selectinload(_Conversation.state)).where(_Conversation.id == conv_id)
            conv = (await session.execute(stmt)).scalar_one_or_none()
            if conv and conv.state and conv.state.state in (_CS.AWAY, _CS.WINDING_DOWN):
                await mgr.update_state(conv_id, new_state=_CS.OBSERVING, is_observing=True)
                logger.info("delayed_return_to_conversation", conversation_id=conv_id)
            return

        # For message/reaction/reconsider actions we need the Telegram chat id
        # and the conversation state (eagerly loaded — async session).
        stmt = _select(_Conversation).options(selectinload(_Conversation.state)).where(_Conversation.id == conv_id)
        conv = (await session.execute(stmt)).scalar_one_or_none()
        if conv is None:
            return
        chat_id = conv.telegram_chat_id

        if atype == "react":
            emoji = (action.payload or {}).get("emoji", "👍")
            target_msg = (action.payload or {}).get("target_message_id")
            if target_msg:
                from telegram import ReactionTypeEmoji
                await self.bot.application.bot.set_message_reaction(
                    chat_id=chat_id, message_id=target_msg,
                    reaction=[ReactionTypeEmoji(emoji=emoji)],
                )
                logger.info("delayed_reaction_sent", conversation_id=conv_id, emoji=emoji)
            return

        if atype in ("reply", "follow_up"):
            text = (action.payload or {}).get("text")
            if not text:
                return

            # Same platform filter as the live send path: hard profanity is
            # bleeped at delivery, the stored payload keeps her raw voice.
            from app.social.censor import censor
            text = censor(text)

            # Follow-ups are "thinking of you" questions — asking them into
            # a chat that has been dead for days feels robotic. Give up if
            # the conversation went cold more than 48h ago.
            if atype == "follow_up" and conv:
                last_active = conv.last_activity_at or (conv.state.updated_at if conv.state else None)
                if last_active and (datetime.utcnow() - last_active).total_seconds() > 48 * 3600:
                    logger.info(
                        "follow_up_dropped_cold_conversation",
                        conversation_id=conv_id,
                    )
                    return

            sent = await self.bot.application.bot.send_message(
                chat_id=chat_id, text=text, disable_web_page_preview=True,
            )
            # Persist Airi's own message with its REAL Telegram id so
            # later reply-to-bot messages resolve the exact message.
            from app.memory.store import get_or_create_user
            bot_user = await get_or_create_user(session, 777000000, self.bot.bot_username or "airi", "Airi", None)
            session.add(_Message(
                conversation_id=conv_id, user_id=bot_user.id,
                telegram_message_id=getattr(sent, "message_id", 0) or 0,
                text=text, timestamp=datetime.utcnow(), is_from_bot=True,
            ))
            logger.info("delayed_message_sent", conversation_id=conv_id, action_type=atype)
            return

        if atype == "reconsider":
            # Cheap reconsideration of a WAIT: did the conversation stay on the
            # thing she noticed, or move on? No LLM — state checks only.
            from app.database import Message as _Msg
            from app.database import ConversationState as _CS
            import random as _random

            # Already chatting? Nothing to come back to.
            if conv.state and conv.state.state == _CS.PARTICIPATING:
                return

            payload = action.payload or {}
            old_topic = (payload.get("topic") or "").lower()
            stmt = (
                _select(_Msg)
                .where(_Msg.conversation_id == conv_id)
                .order_by(_Msg.timestamp.desc())
                .limit(5)
            )
            recent = (await session.execute(stmt)).scalars().all()
            if not recent:
                return  # nothing left to reconsider — conversation moved on

            current_topic = (conv.state.current_topic if conv.state else "") or ""
            moved_on = bool(old_topic) and current_topic.lower() != old_topic
            boredom = (await EmotionalStateManager(session).get_state())["boredom"]
            if moved_on or boredom >= settings.boredom_threshold:
                logger.info("reconsider_expired", conversation_id=conv_id, reason="conversation moved on")
                return

            # Still interesting: she may drift back — but it stays rare, is
            # never a guaranteed message, and never stacks duplicates.
            from app.social.delayed import DelayedActionScheduler
            ds = DelayedActionScheduler(session)
            if await ds.count_pending(conv_id, "return_to_conversation") > 0:
                return
            if boredom < 0.4 and _random.random() < 0.3:
                await ds.schedule_in(
                    conv_id, "return_to_conversation", 5.0,
                    reason="still interested after waiting",
                )
                logger.info("reconsider_kept_interest", conversation_id=conv_id)