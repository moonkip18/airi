from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.llm import llm_provider
from app.language import (
    detect_input_language,
    language_policy,
    LanguageStatsManager,
    DetectedLanguage,
    LanguageDecision,
)
from app.memory import MemoryRetriever, build_retrieval_context, RetrievedMemory
from app.state import (
    ConversationStateManager,
    EmotionalStateManager,
    RelationshipManager,
    ConversationContext,
    WorkingMemoryManager,
    TopicInterestManager,
    SocialGraphManager,
)
from app.social import (
    SocialDecisionEngine, SocialScorer, ParticipationFactors, SocialDecision, SocialAction,
)
from app.social.attention import attention_scorer
from app.social.validation import ResponseValidator, AntiRepetitionTracker
from app.social.delayed import DelayedActionScheduler
from app.analysis import analyze_message, classify_target, MessageTarget, recent_message_hashes
from app.agent.context import ContextBuilder
from app.agent.timeutil import _age_label
from app.agent.learning import SelfModelManager
from app.database import User, Message, ActionType, ConversationState, TrainingSample
from app.config import settings

logger = structlog.get_logger()


class Agent:
    def __init__(
        self,
        session: AsyncSession,
        bot_username: str,
        social_engine: Optional[SocialDecisionEngine] = None,
    ):
        self.session = session
        self.bot_username = bot_username

        self.memory_retriever = MemoryRetriever(session)
        self.conversation_manager = ConversationStateManager(session)
        self.emotional_manager = EmotionalStateManager(session)
        self.relationship_manager = RelationshipManager(session)
        self.context_builder = ContextBuilder(session)
        # Shared engine across messages keeps the rate limiter / burst
        # counters alive; a fresh engine per message would reset them.
        self.social_engine = social_engine or SocialDecisionEngine()
        self.social_scorer = SocialScorer(session)
        self.language_stats = LanguageStatsManager(session)
        self.self_model = SelfModelManager(session)
        self.working_memory = WorkingMemoryManager(session)
        self.topic_interest = TopicInterestManager(session)
        self.social_graph = SocialGraphManager(session)
        self.validator = ResponseValidator(session, bot_username)
        self.anti_repetition = AntiRepetitionTracker(session)
        self.delayed_actions = DelayedActionScheduler(session)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def process_message(
        self,
        conversation_id: int,
        user: User,
        message_text: str,
        is_direct_mention: bool,
        reply_to_user_id: Optional[int] = None,
        telegram_message_id: Optional[int] = None,
        reply_to_message_id: Optional[int] = None,
        mentioned_user_ids: Optional[List[int]] = None,
        message_analysis: Any = None,
        target: Any = None,
        bot_user_id: Optional[int] = None,
        reply_context: Optional[Dict[str, Any]] = None,
        db_message_id: Optional[int] = None,
    ) -> Optional[SocialDecision]:
        from app.database.session import db_write_lock
        # Serialize the pre-LLM phase (reads + state writes + memory
        # retrieval): SQLite's busy_timeout does not cover WAL
        # read->write upgrades, so these sequences need the asyncio
        # lock. It is RELEASED before the LLM call below.
        async with db_write_lock:
            # STEP 0 — message analysis + targeting (cheap, no LLM)
            recent_texts = []
            context0 = await self.conversation_manager.get_context(conversation_id)
            if context0:
                recent_texts = [m.text or "" for m in context0.recent_messages[-10:]]

            if message_analysis is None:
                message_analysis = analyze_message(
                    message_text,
                    message_id=telegram_message_id,
                    speaker_user_id=user.id,
                    speaker_username=user.username,
                    reply_target_message_id=reply_to_message_id,
                    reply_target_user_id=reply_to_user_id,
                    mentioned_user_ids=mentioned_user_ids,
                    current_topic=context0.state.current_topic if context0 else None,
                    recent_hashes=recent_message_hashes(recent_texts),
                    recent_window=recent_texts,
                )

            if target is None:
                target = classify_target(
                    is_direct_mention=is_direct_mention,
                    reply_target_user_id=reply_to_user_id,
                    speaker_user_id=user.id,
                    bot_user_id=bot_user_id,
                    mentioned_user_ids=mentioned_user_ids,
                    text=message_text,
                )

            detected = message_analysis.detected_language or detect_input_language(message_text)

            logger.info(
                "processing_message",
                conversation_id=conversation_id,
                user_id=user.telegram_id,
                username=user.username,
                is_direct_mention=is_direct_mention,
                input_language=detected.language.value,
                target=target.value if hasattr(target, "value") else str(target),
                message_type=message_analysis.message_type,
                novelty=round(message_analysis.novelty, 2),
                humor=round(message_analysis.humor_likelihood, 2),
                text_preview=message_text[:100],
            )

            # STEP 1 — cheap local topic tracking + participant bookkeeping
            # Keep usernames out of the topic space (people are not subjects).
            # Accumulate: every speaker we've ever seen stays protected, not just
            # the current one.
            from app.state.manager import add_usernames_to_blocklist
            add_usernames_to_blocklist([user.username, self.bot_username])
            topic_before = context0.state.current_topic if context0 else None
            await self.conversation_manager.update_topic_local(conversation_id, message_text)
            await self.conversation_manager.increment_other_message(conversation_id)
            await self.conversation_manager.update_participant_activity(conversation_id, user.id)
            await self.conversation_manager.stamp_activity(conversation_id)

            # Ping fatigue (#7): rapid-fire messages from ONE person stack a
            # short-term irritation that time decays. Crossing the threshold
            # nudges her mood once (the crossing is detected from the stored
            # streak, so it survives restarts and per-session agents).
            pulse, fatigue_crossed = await self.relationship_manager.register_message_pulse(user.id)
            if fatigue_crossed:
                await self.emotional_manager.update_from_event("spammed", 1.0)

            # A user replying/typing after our initiation counts as engagement.
            await self.conversation_manager.record_initiation_engagement(conversation_id)

            # STEP 1b — working memory + social observation + jealousy signals
            await self.working_memory.add_message(
                conversation_id, user.id, user.username, message_text,
                is_airi=False,
                reply_target_user_id=reply_to_user_id,
                message_type=message_analysis.message_type,
                importance=0.3,
            )
            if reply_to_user_id and reply_to_user_id != user.id:
                await self.working_memory.add_reply_target(conversation_id, user.id, reply_to_user_id)

            # --- Observed social interaction -------------------------------------
            # Airi watches the whole group, not just messages she participates in.
            # Evidence accumulates gradually in the social graph; a single message
            # never creates a strong bond.
            # Telegram only emits entities for @username syntax — plain names
            # ("alex you're so cute") carry no entity, so resolve textual addressal
            # against the conversation's known people.
            effective_mentions = list(mentioned_user_ids or [])
            if not effective_mentions and context0 is not None:
                from app.memory.retrieval import _resolve_referenced_people
                from app.database import ConversationParticipant
                part_rows = (await self.session.execute(
                    select(ConversationParticipant.user_id).where(
                        ConversationParticipant.conversation_id == conversation_id)
                )).scalars().all()
                effective_mentions = [
                    uid for uid in await _resolve_referenced_people(
                        self.session, message_text, set(part_rows) | {user.id}
                    ) if uid != user.id
                ]

            observed_pair = None
            if reply_to_user_id and reply_to_user_id != user.id:
                observed_pair = (user.id, reply_to_user_id)
            elif effective_mentions:
                # "alex sarah we should queue" — every mentioned person interacts
                # with the speaker.
                observed_pair = (user.id, effective_mentions[0])
            if observed_pair:
                warmth = 0.0
                if message_analysis.humor_likelihood >= 0.4:
                    warmth = 0.5  # banter reads as positive
                elif message_analysis.emotional_salience >= 0.5:
                    warmth = 0.2
                await self.social_graph.observe_interaction(
                    observed_pair[0], observed_pair[1], warmth=warmth,
                    intensity=max(0.3, message_analysis.novelty),
                )
                await self._maybe_jealousy(conversation_id, observed_pair[0], observed_pair[1], message_analysis)

            # --- Affectionate attention toward someone Airi likes -----------------
            # (covers "alex we should queue together ;)" style flirtation without a reply)
            if observed_pair is None and effective_mentions and message_analysis.emotional_salience >= 0.4:
                await self._maybe_jealousy(conversation_id, user.id, effective_mentions[0], message_analysis)

            # STEP 2 — load internal state
            conversation_context = await self.conversation_manager.get_context(conversation_id)
            if not conversation_context:
                logger.warning("no_conversation_context", conversation_id=conversation_id)
                return None

            # Record a topic transition in working memory if the topic changed.
            topic_after = conversation_context.state.current_topic
            if topic_before and topic_after and topic_before.lower() != topic_after.lower():
                await self.conversation_manager.record_topic_transition(conversation_id, topic_before, topic_after)
                await self.working_memory.add_topic_transition(
                    conversation_id, topic_before, topic_after, reason="local_heuristic"
                )

            await self._maybe_rejoin(conversation_context)

            # Did she ask something and get ignored? Humans notice.
            await self._check_being_ignored(conversation_context, is_direct_mention)

            emotional_state = await self.emotional_manager.get_state()
            relationship = await self.relationship_manager.get_relationship_summary(user.id)
            relationships = {}
            for pid in conversation_context.active_users:
                relationships[pid] = await self.relationship_manager.get_relationship_summary(pid)

            # STEP 2b — attention + evolving topic interest
            obsessions = await self.self_model.get_obsessions()
            topic_interests = await self.topic_interest.get_interest_map()
            attention_factors = attention_scorer.score(
                analysis=message_analysis,
                target=target,
                topic_interest=topic_interests,
                relationship=relationship,
                emotional_state=emotional_state,
                current_topic=conversation_context.state.current_topic,
                obsessions=obsessions,
            )
            attention = attention_factors.attention_score

            # Gradually evolve interest in the message's topic from this experience.
            # Use the NORMALIZED topic (same representation as conversation
            # tracking) so filler never becomes a persistent interest.
            from app.state.manager import extract_topic as _extract_topic
            norm_topic, _norm_kw = _extract_topic(
                message_text, session=self.session
            )
            analysis_topic = norm_topic or conversation_context.state.current_topic
            if analysis_topic:
                engaged = attention_factors.topic_interest or message_analysis.emotional_salience or 0.0
                boring = message_analysis.message_type == "short_chatter" and attention < 0.3
                await self.topic_interest.touch(
                    analysis_topic,
                    engagement=max(0.1, min(1.0, engaged)),
                    positive=not boring,
                )

            # STEP 2c — cheap emotional nudges from this event (stateful, inertia).
            if is_direct_mention:
                await self.emotional_manager.update_from_event("direct_mention", 0.5)
            if message_analysis.humor_likelihood >= 0.5:
                await self.emotional_manager.update_from_event("funny", message_analysis.humor_likelihood)
            if message_analysis.message_type == "short_chatter" and attention < 0.3:
                await self.emotional_manager.update_from_event("boring", 1.0)
            if message_analysis.is_question:
                await self.emotional_manager.update_from_event("interesting_topic", 0.3)
            # Refresh emotional state after nudges.
            emotional_state = await self.emotional_manager.get_state()

            # STEP 3 — memory retrieval (semantic + importance + recency + people)
            recent_msgs = [
                {"user_id": msg.user_id, "text": msg.text, "timestamp": msg.timestamp}
                for msg in conversation_context.recent_messages
            ]
            retrieval_context = await build_retrieval_context(
                self.session, conversation_id, user.id, message_text, recent_msgs,
                mentioned_user_ids=mentioned_user_ids,
            )
            retrieved_memories = await self.memory_retriever.retrieve(retrieval_context)

            if settings.verbose_debug:
                logger.debug(
                    "retrieved_memories",
                    conversation_id=conversation_id,
                    count=len(retrieved_memories),
                    memories=[
                        {"type": rm.memory.memory_type.value, "content": rm.memory.content[:80]}
                        for rm in retrieved_memories
                    ],
                )

            # STEP 4 — local participation scoring (NO LLM call), using attention.
            factors = await self.social_scorer.score_message(
                message_text=message_text,
                user_id=user.id,
                conversation_context=conversation_context,
                emotional_state=emotional_state,
                relationship=relationship,
                retrieved_memories=retrieved_memories,
                is_direct_mention=is_direct_mention,
                bot_username=self.bot_username,
                obsessions=obsessions,
                attention=attention,
                message_analysis=message_analysis,
            )

            participation_score = self.social_engine.calculate_participation_score(factors)
            should_participate = self.social_engine.should_participate(
                participation_score, is_direct_mention, False,
                attention=attention,
                boredom=emotional_state.get("boredom", 0.2),
                recent_activity=factors.recent_activity,
                conversation_saturation=factors.conversation_saturation,
                bot_reply_burst=factors.bot_reply_burst,
            )

            logger.info(
                "participation_decision",
                score=round(participation_score, 3),
                attention=round(attention, 3),
                boredom=round(emotional_state.get("boredom", 0.2), 3),
                burst=round(factors.bot_reply_burst, 2),
                should_participate=should_participate,
                direct_mention=is_direct_mention,
                disagreement=factors.is_disagreement,
                mentions_interest=factors.mentions_interest,
                target=target.value if hasattr(target, "value") else str(target),
            )

            # COST GUARD: pure reaction messages ("lol", "😂", "lmaooo") almost
            # never justify an LLM call — a human doesn't verbally answer "lol"
            # either. Handle them entirely on the cheap local path below
            # (reaction / wait / ignore), which carries identical bookkeeping.
            is_bot_reply = bool(reply_context and reply_context.get("is_from_bot"))
            if (
                message_analysis.message_type == "reaction_like"
                and not is_direct_mention
                and not is_bot_reply
                and str(getattr(target, "value", target)) != "directly_to_airi"
            ):
                should_participate = False
                logger.info("llm_skipped_reaction_message", conversation_id=conversation_id)

            # PERCEPTION GUARD: bare media ("[sticker]", "[GIF]", "[photo]" with
            # no caption) is content she literally CANNOT SEE. Spending 30-90s of
            # LLM on it — worse, REACTING to it — breaks the illusion (she'd be
            # laughing at a picture she never saw). She just registers that
            # something was posted and moves on.
            import re as _re
            if _re.fullmatch(r"\[[a-z ]+\]", message_text.strip().lower()) and not is_direct_mention:
                should_participate = False
                logger.info("llm_skipped_bare_media", conversation_id=conversation_id)

            # Boredom can drive an active participant toward leaving.
            want_to_leave = (
                should_participate is False
                and conversation_context.state.is_participating
                and emotional_state.get("boredom", 0.0) >= settings.boredom_leave_threshold
            )

            if not should_participate:
                # A greeting into the room is a social bid, not chatter —
                # acknowledge it with a cheap local reply (no LLM call).
                if (
                    message_analysis.is_greeting
                    and not is_direct_mention
                    and not want_to_leave
                ):
                    greeted = await self._maybe_greet_back(
                        conversation_id, conversation_context, detected,
                    )
                    if greeted is not None:
                        return greeted

                # --- Cheap lightweight reaction path -----------------------------
                # Reactions are far cheaper than speaking and must not require the
                # full reply gate. Not every candidate becomes a reaction, and
                # never two in quick succession.
                reaction_emoji = None
                if not self.social_engine.reaction_on_cooldown(conversation_id):
                    reaction_emoji = self.social_engine.decide_reaction(
                        attention=attention,
                        humor=message_analysis.humor_likelihood,
                        emotional_salience=message_analysis.emotional_salience,
                        boredom=emotional_state.get("boredom", 0.2),
                        recent_activity=factors.recent_activity,
                        conversation_saturation=factors.conversation_saturation,
                        positive=message_analysis.positive,
                        is_question=message_analysis.is_question,
                        conversation_hot=self._conversation_is_hot(conversation_context),
                    )
                if reaction_emoji and not want_to_leave:
                    decision = SocialDecision(
                        action=SocialAction.REACT,
                        reaction_emoji=reaction_emoji,
                        reasoning_summary=f"Mildly worth a reaction (attention {attention:.2f})",
                        current_topic=conversation_context.state.current_topic or "unknown",
                        confidence=0.5,
                        language="english",
                    )
                    self.social_engine.record_reaction(conversation_id)
                    await self._execute_react(conversation_id, decision)
                    await self.working_memory.set_involvement(conversation_id, "observing")
                    await self.working_memory.add_airi_action(
                        conversation_id, "react", {"emoji": reaction_emoji}
                    )
                    await self.emotional_manager.update_from_event("participation", 0.2)
                    logger.info("social_decision_made", action="react",
                                emoji=reaction_emoji, attention=round(attention, 3))
                    return decision

                passive_action = self.social_engine.decide_passive_action(
                    participation_score, attention, emotional_state.get("boredom", 0.2), want_to_leave
                )

                if passive_action == SocialAction.LEAVE:
                    await self._execute_leave(conversation_id)
                    await self.working_memory.set_involvement(conversation_id, "away")
                    return SocialDecision(
                        action=SocialAction.LEAVE,
                        reasoning_summary=f"Bored of the conversation (boredom={emotional_state.get('boredom', 0):.2f})",
                        current_topic=conversation_context.state.current_topic or "unknown",
                        confidence=0.7,
                        continue_participating=False,
                        language="english",
                    )

                await self._handle_silent_observation(
                    conversation_context, emotional_state, user.id, message_text, retrieved_memories, attention
                )

                silent_action = passive_action if passive_action == SocialAction.WAIT else SocialAction.IGNORE
                if silent_action == SocialAction.WAIT:
                    from app.database import ConversationState as _CS
                    if conversation_context.state.state in (_CS.AWAY, _CS.WINDING_DOWN):
                        await self.conversation_manager.update_state(
                            conversation_id, new_state=_CS.OBSERVING, is_observing=True
                        )
                    await self.working_memory.set_involvement(conversation_id, "observing")
                    # WAIT is a real state: she noticed something and may come back
                    # to it. Schedule ONE cheap reconsideration (canceled/inert if
                    # the conversation has moved on by then).
                    await self._schedule_reconsideration(
                        conversation_id, conversation_context, message_analysis, attention
                    )
                else:
                    await self.working_memory.set_involvement(conversation_id, "observing")

                return SocialDecision(
                    action=silent_action,
                    reasoning_summary=(
                        f"WAIT (attention {attention:.2f}, not enough to act)"
                        if silent_action == SocialAction.WAIT
                        else f"Participation score {participation_score:.2f} below threshold"
                    ),
                    current_topic=conversation_context.state.current_topic or "unknown",
                    confidence=1.0 - participation_score,
                    continue_participating=conversation_context.state.is_participating,
                    language="english",
                )

            # STEP 5 — output-language decision (local policy; English default)
            language_decision = await self._decide_language(
                conversation_id, detected, emotional_state, relationship,
                recent_msgs, user,
            )

            # STEP 6 — LLM social decision + response generation
            # BACKLOG GUARD: while she was working through the earlier steps,
            # newer messages may already be waiting (their handlers store them
            # before they queue on the conversation lock). If this message is
            # no longer the freshest thing in the chat — and she wasn't directly
            # addressed — don't spend an LLM call (and seconds of latency)
            # answering a moment that's already gone. She read the room; the
            # freshest message gets the decision.
            if not is_direct_mention and db_message_id is not None:
                from sqlalchemy import func as _func
                newer_count = await self.session.scalar(
                    select(_func.count(Message.id)).where(
                        Message.conversation_id == conversation_id,
                        Message.id > db_message_id,
                        Message.is_from_bot.is_(False),
                    )
                ) or 0
                # The threshold must survive BUSY chats: in a fast group there
                # are always 2-3 newer messages by the time she finishes reading,
                # and the old threshold (2) permanently silenced her during the
                # liveliest moments. A human answering slightly-into-a-burst is
                # normal (the reply thread shows what she's answering); the
                # moment is only truly GONE when a whole pile of new messages
                # buried it. Replies TO her are never stale — that's someone
                # holding the conversation open for her.
                if newer_count >= 4:
                    logger.info(
                        "llm_skipped_backlog",
                        conversation_id=conversation_id,
                        newer_messages=newer_count,
                    )
                    await self.working_memory.set_involvement(conversation_id, "observing")
                    return SocialDecision(
                        action=SocialAction.WAIT,
                        reasoning_summary="chat moved on while she was reading",
                        current_topic=conversation_context.state.current_topic or "unknown",
                        confidence=0.5,
                        continue_participating=conversation_context.state.is_participating,
                        language="english",
                    )

            usernames_by_id = await self._build_usernames_map(conversation_context)
            identity_registry = await self._collect_identities(conversation_context)
            self_model_row = await self.self_model.get_or_create()
            working_summary = await self.working_memory.build_summary(conversation_id)
            person_memories = await self._collect_person_memories(
                conversation_context.active_users,
            )
            # Memory callback (#6): one old, still-relevant memory offered
            # for a natural "remember when" — None most messages.
            memory_callback = await self._pick_callback(retrieved_memories, conversation_id)
            # Autobiographical anchors (#3): what SHE did recently.
            recent_diary = await self._collect_recent_diary()

            # Verbosity streak: rolls (state-weighted) before the prompt is
            # built so the whole reply is written in ONE consistent mode.
            typing_register = await self.conversation_manager.maybe_roll_typing_register(
                conversation_id, emotional_state,
            )

            # Distillation loop: her rated past messages become voice examples.
            await self.context_builder.load_exemplars()
            system_prompt = self.context_builder.build_system_prompt()
            anchor_id = await self.relationship_manager.get_anchor_internal_id()
            context_prompt = self.context_builder.build_context_prompt(
                conversation_context=conversation_context,
                emotional_state=emotional_state,
                relationships=relationships,
                retrieved_memories=retrieved_memories,
                current_user=user,
                is_direct_mention=is_direct_mention,
                message_text=message_text,
                language_decision=language_decision,
                obsessions=obsessions,
                usernames_by_id=usernames_by_id,
                self_model=self_model_row,
                attention=attention,
                attention_explanation=self._attention_explanation(attention_factors),
                target=str(target.value if hasattr(target, "value") else target),
                message_analysis=message_analysis,
                working_memory_summary=working_summary,
                topic_interests=topic_interests,
                social_graph=await self.social_graph.get_edge_summaries(conversation_context.active_users),
                anchor_user_id=anchor_id,
                reply_context=reply_context,
                identities=identity_registry,
                person_memories=person_memories,
                typing_register=typing_register,
                memory_callback=memory_callback,
                recent_diary=recent_diary,
            )

            messages_payload = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_prompt},
            ]

            # Release the SQLite write lock BEFORE the LLM call: everything above
            # is committed so the (possibly slow) generation below never holds
            # the database hostage. Post-decision updates run in a new transaction.
            from app.database import commit_with_retry
            await commit_with_retry(self.session)

        try:
            decision = await llm_provider.generate_structured(
                messages_payload, SocialDecision,
                temperature=self._decision_temperature(emotional_state),
                # Generous budget: the thinking phase (when not disabled)
                # counts toward max_tokens; a truncated reply costs a full
                # retry round-trip, which is worse than an unused cap.
                max_tokens=2000,
            )
            decision.language = language_decision.language
            decision.response_delay = self.social_engine.calculate_response_delay(
                is_direct_mention=is_direct_mention,
                is_autonomous=False,
                message_complexity=min(1.0, len(message_text) / 300.0),
                conversation_pace=conversation_context.state.activity_level,
                energy=emotional_state.get("energy", 0.5),
            )

            # Post-decision writes run under the global write lock and commit
            # INSIDE it (an uncommitted write tx released from the lock holds
            # SQLite's RESERVED lock and blocks every other writer for the
            # full busy_timeout — the flush-then-commit-later pattern is what
            # caused the "database is locked" cascades).
            from app.database.session import db_write_lock as _dbwl, commit_with_retry as _cwr
            async with _dbwl:
                await self._execute_decision(
                    conversation_id, decision, retrieved_memories, user.id,
                    emotional_state, relationship, language_decision,
                    conversation_context, attention,
                    message_analysis=message_analysis,
                )

                await self._update_conversation_state(conversation_id, decision, conversation_context)

                if decision.response:
                    await self._record_training_sample(
                        conversation_id, "reply", decision,
                        conversation_context, usernames_by_id,
                    )
                await _cwr(self.session)

            logger.info(
                "social_decision_made",
                action=decision.action.value,
                confidence=decision.confidence,
                attention=round(attention, 3),
                language=decision.language,
                reasoning=decision.reasoning_summary[:100],
            )

            return decision

        except Exception as e:
            logger.error("llm_decision_failed", error=str(e))
            return SocialDecision(
                action=SocialAction.IGNORE,
                reasoning_summary=f"LLM error: {str(e)}",
                current_topic=conversation_context.state.current_topic or "unknown",
                confidence=0.0,
                continue_participating=False,
                language="english",
            )

    def _attention_explanation(self, af) -> str:
        parts = []
        if af.topic_interest >= 0.6:
            parts.append("topic genuinely interests you")
        if af.directness >= 0.8:
            parts.append("the message is directed at you")
        if af.humor >= 0.5:
            parts.append("it's funny")
        if af.emotional_salience >= 0.5:
            parts.append("there's real emotion in it")
        if af.conflict >= 0.5:
            parts.append("it's a challenge or argument")
        if af.boredom >= 0.6:
            parts.append("you're bored right now")
        return "you're paying attention because " + ", ".join(parts) if parts else "low attention — this barely registers"

    def _decision_temperature(self, emotional_state: Dict[str, float]) -> float:
        """state-driven chaos dial (no fine-tuning needed): LLM sampling
        temperature modulated by her emotional state. Amused/bored/competitive
        moments sample hotter (more chaotic, unpredictable output);
        annoyed/embarrassed moments sample cooler (blunt and careful, not
        random). Bounded by DECISION_TEMPERATURE_* settings."""
        base = settings.decision_temperature_base
        heat = (
            0.18 * max(0.0, emotional_state.get("amusement", 0.0) - 0.4)
            + 0.10 * max(0.0, emotional_state.get("boredom", 0.2) - 0.3)
            + 0.08 * max(0.0, emotional_state.get("competitiveness", 0.5) - 0.6)
            + 0.05 * max(0.0, emotional_state.get("energy", 0.5) - 0.7)
        )
        chill = (
            0.08 * emotional_state.get("annoyance", 0.0)
            + 0.06 * emotional_state.get("embarrassment", 0.0)
        )
        # Rounded to 2 decimals: B.AI's upstream GLM rejects temperatures
        # with more precision ("temperature参数非法：限制小数点[2]位"), and
        # the float arithmetic above produces long tails (0.5468000...1).
        return round(max(0.35, min(settings.decision_temperature_max, base + heat - chill)), 2)

    # The filter is part of her character, not a silent censor: when the
    # deterministic validator hard-blocks a reply, she occasionally just
    # says it — human-style. Kept rare via probability + anti-repetition.
    FILTERED_LINES = [
        "filtered",
        "filtered lol",
        "nope. filtered",
        "that one got filtered",
    ]

    async def _record_training_sample(
        self,
        conversation_id: int,
        trigger: str,
        decision: SocialDecision,
        context: ConversationContext,
        usernames_by_id: Dict[int, str],
    ):
        """Distillation-loop step 1: collect. Persist every reply she actually
        sends together with a compact snapshot of the conversation that
        produced it. Later: human-curate (rating column) -> distill into
        few-shot exemplars or fine-tune. Pure bookkeeping, no LLM cost."""
        if not settings.training_sample_logging:
            return
        try:
            bot_label = (self.bot_username or "airi").lower()
            snapshot = [
                {
                    "sender": (
                        bot_label if msg.is_from_bot
                        else usernames_by_id.get(msg.user_id, f"user{msg.user_id}")
                    ),
                    "text": (msg.text or "")[:200],
                }
                for msg in context.recent_messages[-10:]
                if msg.text
            ]
            self.session.add(TrainingSample(
                conversation_id=conversation_id,
                trigger=trigger,
                context_snapshot=snapshot,
                response_text=(decision.response or "")[:1000],
                action=(
                    decision.action.value
                    if hasattr(decision.action, "value")
                    else str(decision.action)
                ),
                language=decision.language or "english",
            ))
        except Exception as e:
            logger.warning("training_sample_record_failed", error=str(e))

    async def _schedule_reconsideration(
        self,
        conversation_id: int,
        context: ConversationContext,
        analysis: Any,
        attention: float,
    ):
        """WAIT made real: one lightweight, cancelable future re-check.

        The scheduled action is executed by the background loop with CHEAP
        state checks only (no LLM). If the conversation has moved on, it
        silently expires.

        Deduplicated: never more than ONE outstanding reconsider per
        conversation — active chats would otherwise schedule one per
        held-back message.
        """
        if await self.delayed_actions.count_pending(conversation_id, "reconsider") > 0:
            return

        await self.delayed_actions.schedule_in(
            conversation_id,
            "reconsider",
            delay_seconds=60.0,
            payload={
                "topic": context.state.current_topic,
                "attention": round(attention, 2),
                "user_id": analysis.speaker_user_id if analysis else None,
            },
            reason="noticed something, deciding whether it's worth coming back to",
        )

    # Affectionate/flirty cues that suggest someone is directing unusual warmth
    # at another person. Used only as *candidate* jealousy signals.
    AFFECTION_CUES = ("❤", "😍", "😘", "();", ";)", "love you", "ily", "cutie", "bestie", "marry")

    async def _maybe_jealousy(
        self,
        conversation_id: int,
        actor_id: int,
        receiver_id: int,
        analysis: Any,
    ):
        """Derive a jealousy signal from a real social event — never randomly.

        Jealousy is only possible when one of the two people involved is
        someone Airi is strongly attached to (she can be jealous about the
        person she likes *receiving* attention, or about them *giving* it).
        The social graph must confirm a bond, her state must allow it, and
        even then it's probabilistic and mild.
        """
        import random as _random
        text = (analysis.raw_text or "").lower() if analysis else ""
        if not any(cue in text for cue in self.AFFECTION_CUES):
            return  # ordinary conversation — no jealousy, ever

        # Who (if anyone) in this exchange does Airi care about?
        liked_id = None
        liked_rel = None
        for person in (actor_id, receiver_id):
            rel = await self.relationship_manager.get_relationship_summary(person)
            if rel.get("affection", 0.0) >= 0.45:
                liked_id, liked_rel = person, rel
                break
        if liked_id is None:
            return  # weak/no attachment -> jealousy unlikely

        # Repeated evidence: the pair must have some observed bond already.
        closeness = await self.social_graph.closeness_between(actor_id, receiver_id)
        if closeness < 0.03:
            return

        emotional = await self.emotional_manager.get_state()
        base = 0.25 * liked_rel.get("affection", 0.0) + 0.15 * closeness
        base *= (0.5 + emotional.get("interest", 0.5) * 0.5)
        base *= (1.0 - emotional.get("boredom", 0.2) * 0.4)

        if _random.random() >= base:
            return

        await self.emotional_manager.update_from_event("jealousy_trigger", 0.8)
        await self.relationship_manager.update_from_interaction(liked_id, "jealousy_trigger", 0.6)
        await self.working_memory.add_event(
            conversation_id, "jealousy", user_id=actor_id,
            payload={"about": receiver_id, "cue": text[:60]}, importance=0.7,
        )
        logger.info(
            "jealousy_signal",
            conversation_id=conversation_id,
            about_user=liked_id,
            cue=text[:40],
        )

    # ------------------------------------------------------------------
    # Language decision
    # ------------------------------------------------------------------

    async def _decide_language(
        self,
        conversation_id: int,
        detected: DetectedLanguage,
        emotional_state: Dict[str, float],
        relationship: Dict[str, float],
        recent_msgs: List[Dict[str, Any]],
        user: User,
    ) -> LanguageDecision:
        recent_persian = await self.language_stats.get_recent_persian_usage(conversation_id)
        recurring_jokes = await self.self_model.get_recurring_jokes()
        decision = language_policy.decide(
            detected=detected,
            emotional_state=emotional_state,
            relationship=relationship,
            recent_messages=recent_msgs,
            recurring_jokes=recurring_jokes,
            recent_persian_outputs=recent_persian,
        )
        logger.info(
            "language_decision",
            input_language=detected.language.value,
            output_language=decision.language,
            reason=decision.reason,
            allow_persian_snippet=decision.allow_persian_snippet,
        )
        return decision

    @staticmethod
    def _conversation_is_hot(context: ConversationContext) -> bool:
        """3+ messages in the last 3 minutes = the conversation is actively
        moving. That's ANSWER territory, not reaction territory."""
        cutoff = datetime.utcnow() - timedelta(minutes=3)
        recent = context.recent_messages[-5:]
        hot = sum(
            1 for m in recent
            if m.timestamp and m.timestamp > cutoff
        )
        return hot >= 3

    BOT_SENTINEL_TELEGRAM_ID = 777000000

    async def _collect_identities(self, context: ConversationContext) -> Dict[int, Dict[str, Any]]:
        """Canonical identity registry: internal id -> identity info.

        Telegram user ID is the canonical identity. Usernames/display names
        are mutable attributes. The bot's own messages are identified by the
        sentinel telegram id, never by guessing from names.
        """
        from sqlalchemy import select as _select

        ids = {m.user_id for m in context.recent_messages} | set(context.active_users)
        rows: Dict[int, Dict[str, Any]] = {}
        for uid in ids:
            if uid is None:
                continue
            result = await self.session.execute(_select(User).where(User.id == uid))
            db_user = result.scalar_one_or_none()
            if db_user is None:
                continue
            is_bot = db_user.telegram_id == self.BOT_SENTINEL_TELEGRAM_ID
            name = (self.bot_username or "Airi") if is_bot else (
                db_user.username or db_user.first_name or f"user{db_user.telegram_id}"
            )
            rows[uid] = {
                "name": name,
                "username": db_user.username or "",
                "first_name": db_user.first_name or "",
                "telegram_id": db_user.telegram_id,
                "is_bot": is_bot,
            }
        return rows

    async def _collect_person_memories(
        self,
        active_user_ids,
        limit_per: int = 2,
    ) -> Dict[int, List[str]]:
        """Per-person memory dossier: each active person's OWN most important
        memories, rendered under their name in the context. This is the
        anti-contamination fix - facts stay glued to the person they're
        about instead of floating in a shared memory soup.

        Time-aware (#2): each fact carries a human-scale age ("2d ago",
        "3w ago") — people remember WHEN they learned things, and "you
        said X 3 weeks ago" is what continuity actually sounds like."""
        from sqlalchemy import select as _select, desc as _desc
        from app.database import Memory as _Memory

        out: Dict[int, List[str]] = {}
        for pid in list(active_user_ids)[:6]:
            if pid is None:
                continue
            rows = (await self.session.execute(
                _select(_Memory)
                .where(
                    _Memory.user_id == pid,
                    _Memory.superseded_by.is_(None),
                )
                .order_by(_desc(_Memory.importance), _desc(_Memory.created_at))
                .limit(limit_per)
            )).scalars().all()
            snippets = [
                f"{(m.content or '').strip()[:140]} ({_age_label(m.created_at)})"
                for m in rows
                if (m.content or "").strip()
            ]
            if snippets:
                out[pid] = snippets
        return out

    async def _pick_callback(self, retrieved_memories, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Memory callback (#6): resurface ONE old episodic/relationship
        memory when it's relevant to RIGHT NOW but old enough that 'remember
        when' framing is natural (not something from an hour ago).

        Selection rules:
          * only episodic / relationship types (facts don't make stories)
          * age >= settings.callback_min_age_days
          * refractory: a callback is not re-used for at least
            settings.callback_refractory_days (persisted in recent_phrases,
            so restarts can't resurrect the same story on loop)
        Returns {text, age_label, memory_id} or None. The PROMPT only
        offers it; whether she uses it stays her decision (silence first).
        """
        from app.database import MemoryType as _MT
        from app.database import RecentPhrase as _RP

        now = datetime.utcnow()
        best = None
        best_score = 0.0
        for rm in retrieved_memories:
            m = rm.memory
            if m.memory_type not in (_MT.EPISODIC, _MT.RELATIONSHIP):
                continue
            created = m.created_at
            if not created:
                continue
            age_days = (now - created).total_seconds() / 86400.0
            if age_days < settings.callback_min_age_days:
                continue
            # Relevance * age-boost: old memories that STILL retrieve HIGH
            # are exactly the "we keep coming back to this" stories humans
            # retell. Recency-decayed repeats rank below fresh retells.
            score = float(rm.relevance_score or 0.0) * min(1.0, 0.4 + age_days / 30.0)
            if score < settings.callback_min_relevance:
                continue
            if m.last_retrieved:
                since = (now - m.last_retrieved).total_seconds() / 86400.0
                if since < settings.callback_refractory_days:
                    continue
            if score > best_score:
                best, best_score = m, score
        if best is None:
            return None

        # Persisted refractory: did we already surface this story recently?
        refractory_cutoff = now - timedelta(days=settings.callback_refractory_days)
        recently_used = await self.session.scalar(
            select(func.count(_RP.id)).where(
                _RP.phrase_type == "callback",
                _RP.conversation_id == conversation_id,
                _RP.content == (best.content or "")[:300],
                _RP.created_at > refractory_cutoff,
            )
        )
        if recently_used:
            return None

        await self.anti_repetition.record(
            conversation_id, "callback", best.content or "",
        )
        return {
            "memory_id": best.id,
            "text": (best.content or "").strip()[:200],
            "age_label": _age_label(best.created_at),
        }

    async def _collect_recent_diary(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Autobiographical continuity (#3): her own recent EPISODIC
        memories — things she DID, said, promised, played. She currently
        remembers everyone but herself; these self-anchors are what
        'i was grinding p5 all night' continuity comes from.

        Ownership is the join: extraction attributes facts about the
        persona to her user row (via about_username), so memories with
        user_id == the bot's internal id ARE her autobiography."""
        from sqlalchemy import select as _select, desc as _desc
        from app.database import Memory as _Memory, MemoryType as _MT, User as _User

        bot_row = (await self.session.execute(
            _select(_User).where(_User.telegram_id == self.BOT_SENTINEL_TELEGRAM_ID)
        )).scalar_one_or_none()
        if bot_row is None:
            return []

        rows = (await self.session.execute(
            _select(_Memory)
            .where(
                _Memory.user_id == bot_row.id,
                _Memory.memory_type == _MT.EPISODIC,
                _Memory.superseded_by.is_(None),
            )
            .order_by(_desc(_Memory.created_at))
            .limit(limit)
        )).scalars().all()
        return [
            {
                "text": (m.content or "").strip()[:180],
                "age_label": _age_label(m.created_at),
            }
            for m in rows
            if (m.content or "").strip()
        ]

    async def _build_usernames_map(self, context: ConversationContext) -> Dict[int, str]:
        """internal user id -> unique display label (collision-proof).

        Two people who share a username/display name get distinct labels
        (suffixed with their immutable Telegram id), so the LLM can never
        treat them as one person.
        """
        rows = await self._collect_identities(context)

        counts: Dict[str, int] = {}
        for info in rows.values():
            key = info["name"].strip().lower()
            counts[key] = counts.get(key, 0) + 1

        labels: Dict[int, str] = {}
        for uid, info in rows.items():
            name = info["name"]
            if counts[name.strip().lower()] > 1:
                labels[uid] = f"{name} (id {info['telegram_id']})"
            else:
                labels[uid] = name
        return labels

    # ------------------------------------------------------------------
    # Conversation lifecycle
    # ------------------------------------------------------------------

    async def _maybe_rejoin(self, context: ConversationContext):
        """Return to OBSERVING after being away/wound-down past the cooldown."""
        state = context.state
        if state.state.value not in ("away", "winding_down"):
            return

        last_activity = context.conversation.last_activity_at or state.updated_at
        if last_activity is None:
            return

        cooldown = timedelta(seconds=settings.return_cooldown_seconds)
        if datetime.utcnow() - last_activity > cooldown:
            from app.database import ConversationState
            await self.conversation_manager.update_state(
                context.conversation.id,
                new_state=ConversationState.OBSERVING,
                is_observing=True,
                is_participating=False,
                winding_down_score=0.0,
                initiations_without_engagement=0,
            )
            await self.emotional_manager.update_from_event("returned")
            logger.info("airi_returned", conversation_id=context.conversation.id)

    # Local greeting replies, per input language. Formulaic by design —
    # no LLM call needed to say hi back.
    GREETINGS = {
        "persian": ["سلام", "سلام سلام", "هی"],
        "english": ["yo", "hi", "hey", "hellooo"],
    }

    async def _maybe_greet_back(
        self,
        conversation_id: int,
        context: ConversationContext,
        detected: DetectedLanguage,
    ) -> Optional[SocialDecision]:
        """Greet the room back — locally, no LLM.

        Humans don't answer every salam either: one local greeting per
        conversation per ~45 minutes, probabilistic, and never into a
        mention (mentions get the full LLM reply).
        """
        import random as _random
        from datetime import datetime as _dt

        if self.social_engine.is_rate_limited():
            return None

        # Rate limit via working memory: one greet per conversation/45min.
        recent_actions = await self.working_memory.get_airi_actions(
            conversation_id, limit=10,
        )
        for action_row in recent_actions:
            if action_row.payload.get("action") == "greet":
                at = action_row.payload.get("at")
                try:
                    greeted_at = _dt.fromisoformat(at)
                    if (_dt.utcnow() - greeted_at).total_seconds() < 45 * 60:
                        return None
                except Exception:
                    pass

        if _random.random() > 0.6:
            return None

        lang = (
            "persian" if detected.language.value == "persian" else "english"
        )
        text = _random.choice(self.GREETINGS[lang])

        # Same accounting as a real reply.
        await self.conversation_manager.increment_bot_message(conversation_id)
        self.social_engine.record_message()
        await self.anti_repetition.record_response(
            conversation_id, text, "", target_user_id=None,
        )
        await self.working_memory.add_airi_action(
            conversation_id, "greet", {"text": text},
        )
        await self.emotional_manager.update_from_event("participation", 0.3)

        logger.info(
            "greeted_back",
            conversation_id=conversation_id,
            text=text,
        )

        return SocialDecision(
            action=SocialAction.REPLY,
            reasoning_summary="greeting the room back",
            current_topic=context.state.current_topic or "casual",
            confidence=0.6,
            response=text,
            continue_participating=True,
            language=lang,
            response_delay=_random.uniform(0.8, 2.0),
        )

    async def _check_being_ignored(
        self,
        context: ConversationContext,
        is_direct_mention: bool,
    ):
        """She asked a question, then several messages passed with no answer.

        Fires exactly once per unanswered question (when the 4th message
        lands): a mood hit via the 'ignored' emotional event and a working
        memory entry she can actually refer to ("ok ignore me then").
        """
        if is_direct_mention:
            return

        msgs = context.recent_messages
        last_bot_idx = None
        for i, m in enumerate(msgs):
            if m.is_from_bot:
                last_bot_idx = i

        if last_bot_idx is None:
            return

        after = msgs[last_bot_idx + 1:]
        if len(after) != 4:
            return  # fires only at the moment the threshold is crossed

        bot_msg = msgs[last_bot_idx]
        if "?" not in (bot_msg.text or ""):
            return  # she didn't ask anything — silence is just the flow

        bot_internal_id = bot_msg.user_id
        if any(m.reply_to_user_id == bot_internal_id for m in after):
            return  # someone did answer her

        await self.emotional_manager.update_from_event("ignored", 0.8)
        await self.working_memory.add_event(
            context.conversation.id, "ignored",
            user_id=after[-1].user_id, importance=0.6,
        )
        logger.info(
            "airi_ignored",
            conversation_id=context.conversation.id,
            question_preview=(bot_msg.text or "")[:60],
        )

    async def _handle_silent_observation(
        self,
        context: ConversationContext,
        emotional_state: Dict[str, float],
        user_id: int,
        message_text: str,
        retrieved_memories: List[RetrievedMemory],
        attention: float = 0.0,
    ):
        # While quiet she still notices things: interesting messages can pull
        # her from AWAY into OBSERVING.
        if context.state.state.value == "away":
            if self._is_interesting(message_text, retrieved_memories, emotional_state, attention):
                from app.database import ConversationState
                await self.conversation_manager.update_state(
                    context.conversation.id,
                    new_state=ConversationState.OBSERVING,
                    is_observing=True,
                )

    def _is_interesting(self, message_text: str, memories: List[RetrievedMemory], emotional_state: Dict[str, float], attention: float = 0.0) -> bool:
        if attention >= 0.6:
            return True
        if emotional_state.get("curiosity", 0.5) > 0.7:
            return True
        if any(m.relevance_score > 0.7 for m in memories):
            return True
        return False

    def _room_avg_words(self, context: ConversationContext) -> float:
        """Average words per recent HUMAN message — is the room in
        one-liner mode or long-form mode right now? Feeds the casual
        length gate in the validator (she matched the room, or she's the
        essay-person in a one-liner room)."""
        texts = [
            (m.text or "")
            for m in context.recent_messages[-6:]
            if not m.is_from_bot and m.text
        ]
        if not texts:
            return 0.0
        return sum(len(t.split()) for t in texts) / len(texts)


    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    async def _execute_decision(
        self,
        conversation_id: int,
        decision: SocialDecision,
        retrieved_memories: List[RetrievedMemory],
        user_id: int,
        emotional_state: Dict[str, float],
        relationship: Dict[str, float],
        language_decision: LanguageDecision,
        context: ConversationContext,
        attention: float,
        message_analysis: Any = None,
    ):
        # Lightweight deterministic response validation (no extra LLM call).
        if decision.response and decision.action in (SocialAction.REPLY, SocialAction.INITIATE):
            validated = await self.validator.validate(
                decision.response,
                current_topic=context.state.current_topic,
                is_reply_to_airi=True,
                important_memories=[rm.memory for rm in retrieved_memories if rm.memory.importance >= 0.7],
                recent_topic_history=[str(t.get("new_topic", "")) for t in (context.state.recent_topics or [])],
                conversation_id=conversation_id,
                target_user_id=user_id,
                room_avg_words=self._room_avg_words(context),
            )
            if await self.validator.should_skip(validated):
                logger.info(
                    "response_blocked_by_validation",
                    issues=validated["issues"],
                    response_preview=(decision.response or "")[:60],
                )
                await self.anti_repetition.record(conversation_id, "response_pattern", decision.response or "",
                                                  target_user_id=user_id)
                decision.action = SocialAction.WAIT
                decision.response = None
                # The filter is visible character, not a silent censor: she
                # OCCASIONALLY says a meta-line instead of vanishing — but
                # only when the blocked reply was (or looked like) a JOKE.
                # A blocked substantive answer (a real question was asked)
                # must not become "filtered" — that's a contentless reply to
                # a real person asking a real thing (the 14:04:57 artifact).
                import random as _random
                joke_like = bool(message_analysis and (
                    getattr(message_analysis, "message_type", "") == "joke"
                    or getattr(message_analysis, "humor", 0) >= 0.5
                ))
                if joke_like and _random.random() < settings.filtered_reply_probability:
                    line = _random.choice(self.FILTERED_LINES)
                    if not await self.anti_repetition.is_recent_duplicate(
                        conversation_id, line, target_user_id=user_id,
                        topic=context.state.current_topic,
                    ):
                        decision.action = SocialAction.REPLY
                        decision.response = line
                        decision.reasoning_summary = "blocked joke reply -> filtered meta-line"
            else:
                decision.response = validated["response"]

        if decision.action == SocialAction.REPLY:
            await self._execute_reply(
                conversation_id, decision, retrieved_memories, user_id,
                emotional_state, relationship, language_decision,
            )
            await self.working_memory.set_involvement(conversation_id, "participating")
        elif decision.action == SocialAction.REACT:
            # The LLM can also pick REACT — enforce the same cooldown so it
            # never stacks on the local path's reactions.
            if self.social_engine.reaction_on_cooldown(conversation_id):
                decision.action = SocialAction.WAIT
                decision.reaction_emoji = None
                await self.working_memory.set_involvement(conversation_id, "observing")
                return
            self.social_engine.record_reaction(conversation_id)
            await self._execute_react(conversation_id, decision)
            await self.working_memory.set_involvement(conversation_id, "observing")
        elif decision.action == SocialAction.LEAVE:
            await self._execute_leave(conversation_id)
            await self.working_memory.set_involvement(conversation_id, "away")
        elif decision.action == SocialAction.WAIT:
            await self.working_memory.set_involvement(conversation_id, "observing")
        # INITIATE is handled through the autonomous path.

    async def _execute_react(self, conversation_id: int, decision: SocialDecision):
        # The LLM sometimes picks an emoji Telegram doesn't allow as a
        # reaction (😂, 💀...) — swap it for a valid one instead of failing.
        if not decision.reaction_emoji or decision.reaction_emoji not in self.social_engine.VALID_REACTIONS:
            import random as _random
            decision.reaction_emoji = _random.choice(
                self.social_engine.REACTION_EMOJIS_FUNNY
                + self.social_engine.REACTION_EMOJIS_WOW
            )
        logger.info("airi_reacts", conversation_id=conversation_id, emoji=decision.reaction_emoji)

    async def _execute_reply(
        self,
        conversation_id: int,
        decision: SocialDecision,
        retrieved_memories: List[RetrievedMemory],
        user_id: int,
        emotional_state: Dict[str, float],
        relationship: Dict[str, float],
        language_decision: LanguageDecision,
    ):
        if self.social_engine.is_rate_limited():
            logger.warning("reply_blocked_rate_limit", conversation_id=conversation_id)
            return

        await self.conversation_manager.increment_bot_message(conversation_id)
        self.social_engine.record_message()

        if decision.memory_ids_used:
            for mem_uuid in decision.memory_ids_used:
                memory = await self.memory_retriever.store.get_memory_by_uuid(mem_uuid)
                if memory:
                    memory.last_retrieved = datetime.utcnow()
                    memory.retrieval_count += 1

        # Language accounting (keeps English-dominance measurable)
        has_snippet = bool(language_decision.allow_persian_snippet and decision.language == "english")
        await self.language_stats.record_response(
            conversation_id,
            decision.language or "english",
            language_decision.reason,
            has_snippet,
        )

        # Anti-repetition: remember what she said (with target/context) so she varies.
        await self.anti_repetition.record_response(
            conversation_id, decision.response or "", decision.current_topic,
            target_user_id=user_id,
        )
        await self.working_memory.add_airi_action(
            conversation_id, "reply", {"text": (decision.response or "")[:200], "topic": decision.current_topic}
        )

        await self.emotional_manager.update_from_event(
            "participation", 0.5, user_relationship=relationship,
        )
        await self.relationship_manager.update_from_interaction(user_id, "positive", 0.5)

    async def _execute_leave(self, conversation_id: int):
        await self.conversation_manager.update_state(
            conversation_id,
            new_state=ConversationState.WINDING_DOWN,
            is_participating=False,
            winding_down_score=1.0,
        )
        await self.emotional_manager.update_from_event("left_conversation")
        # Let her return later: schedule a natural re-entry nudge.
        await self.delayed_actions.schedule_in(
            conversation_id, "return_to_conversation", settings.return_cooldown_seconds,
            reason="left due to boredom; may drift back later",
        )
        logger.info("airi_left_conversation", conversation_id=conversation_id)


    async def _update_conversation_state(
        self,
        conversation_id: int,
        decision: SocialDecision,
        context: ConversationContext,
    ):
        new_state = context.state.state

        if decision.action == SocialAction.REPLY:
            if context.state.state in [ConversationState.AWAY, ConversationState.OBSERVING, ConversationState.INTERESTED]:
                new_state = ConversationState.PARTICIPATING

        elif decision.action == SocialAction.LEAVE:
            new_state = ConversationState.WINDING_DOWN

        elif decision.action in (SocialAction.WAIT, SocialAction.REACT):
            if context.state.state in (ConversationState.AWAY, ConversationState.WINDING_DOWN):
                new_state = ConversationState.OBSERVING

        elif decision.action == SocialAction.IGNORE:
            if context.state.state == ConversationState.PARTICIPATING:
                winding_down = await self.conversation_manager.detect_winding_down(conversation_id)
                if winding_down > 0.7:
                    new_state = ConversationState.WINDING_DOWN

        await self.conversation_manager.update_state(
            conversation_id,
            new_state=new_state if new_state != context.state.state else None,
            is_participating=decision.continue_participating and decision.action == SocialAction.REPLY,
            last_bot_action=ActionType(decision.action.value) if decision.action != SocialAction.IGNORE else None,
        )

    # ------------------------------------------------------------------
    # Autonomous participation
    # ------------------------------------------------------------------

    async def autonomous_check(self, conversation_id: int) -> Optional[SocialDecision]:
        """Autonomous participation: Airi may enter a conversation uninvited.

        Only when enabled, not rate-limited, currently away/observing, the
        topic genuinely interests her, and she hasn't spammed failed
        initiations. Produces a REAL initiation message via the LLM
        (decision first, generation second).
        """
        if not settings.autonomous_behavior_enabled:
            return None

        if self.social_engine.is_rate_limited():
            return None

        context = await self.conversation_manager.get_context(conversation_id)
        if not context or context.state.state not in [ConversationState.AWAY, ConversationState.OBSERVING]:
            return None

        # Don't nag: stop initiating after repeated failures to engage.
        if context.state.initiations_without_engagement >= 2:
            return None

        # Cooldown between initiations.
        if context.state.last_initiation_at:
            if datetime.utcnow() - context.state.last_initiation_at < timedelta(minutes=20):
                return None

        recent_messages = context.recent_messages[-5:]
        if not recent_messages:
            return None

        last_msg = recent_messages[-1]
        if last_msg.is_from_bot:
            return None


        emotional_state = await self.emotional_manager.get_state()
        obsessions = await self.self_model.get_obsessions()
        relationship = await self.relationship_manager.get_relationship_summary(last_msg.user_id)
        topic_interests = await self.topic_interest.get_interest_map()

        # Attention on the latest message — gates whether she's genuinely pulled in.
        analysis = analyze_message(
            last_msg.text or "",
            speaker_user_id=last_msg.user_id,
            current_topic=context.state.current_topic,
        )
        attention_factors = attention_scorer.score(
            analysis=analysis,
            target=MessageTarget.GENERAL_GROUP_MESSAGE,
            topic_interest=topic_interests,
            relationship=relationship,
            emotional_state=emotional_state,
            current_topic=context.state.current_topic,
            obsessions=obsessions,
        )
        attention = attention_factors.attention_score
        if attention < settings.attention_participation_threshold:
            return None

        retrieval_context = await build_retrieval_context(
            self.session,
            conversation_id,
            last_msg.user_id,
            last_msg.text or "",
            [
                {"user_id": msg.user_id, "text": msg.text, "timestamp": msg.timestamp}
                for msg in context.recent_messages
            ],
        )
        retrieved_memories = await self.memory_retriever.retrieve(retrieval_context)

        factors = await self.social_scorer.score_message(
            message_text=last_msg.text or "",
            user_id=last_msg.user_id,
            conversation_context=context,
            emotional_state=emotional_state,
            relationship=relationship,
            retrieved_memories=retrieved_memories,
            is_direct_mention=False,
            bot_username=self.bot_username,
            obsessions=obsessions,
            attention=attention,
            message_analysis=analysis,
        )
        factors.is_autonomous = True

        score = self.social_engine.calculate_participation_score(factors)
        # Uninvited participation is deliberately harder than replying: the
        # gate sits ABOVE the normal participation bar (threshold default
        # 0.65 -> effective 0.45, reachable only by messages that genuinely
        # grab her). Combined with the attention gate and cooldowns below,
        # this keeps autonomous joins rare and natural.
        autonomous_gate = max(0.0, settings.autonomous_participation_threshold - 0.20)
        if score < autonomous_gate:
            return None

        # She decided to jump in — build context and generate the initiation.
        detected = detect_input_language(last_msg.text or "")
        language_decision = await self._decide_language(
            conversation_id, detected, emotional_state, relationship,
            [{"user_id": m.user_id, "text": m.text} for m in context.recent_messages],
            user=None,
        )
        usernames_by_id = await self._build_usernames_map(context)
        identity_registry = await self._collect_identities(context)
        self_model_row = await self.self_model.get_or_create()
        working_summary = await self.working_memory.build_summary(conversation_id)

        # Same verbosity streak as the reply pipeline — an initiation is
        # written in whatever mode she's currently in.
        typing_register = await self.conversation_manager.maybe_roll_typing_register(
            conversation_id, emotional_state,
        )

        system_prompt = self.context_builder.build_system_prompt()
        context_prompt = self.context_builder.build_context_prompt(
            conversation_context=context,
            emotional_state=emotional_state,
            relationships={last_msg.user_id: relationship},
            retrieved_memories=retrieved_memories,
            current_user=None,
            is_direct_mention=False,
            message_text=last_msg.text or "",
            language_decision=language_decision,
            obsessions=obsessions,
            usernames_by_id=usernames_by_id,
            self_model=self_model_row,
            autonomous=True,
            attention=attention,
            attention_explanation=self._attention_explanation(attention_factors),
            working_memory_summary=working_summary,
            message_analysis=analysis,
            topic_interests=topic_interests,
            social_graph=await self.social_graph.get_edge_summaries(context.active_users),
            identities=identity_registry,
            typing_register=typing_register,
        )

        # Release the DB write lock before generation (see process_message).
        from app.database import commit_with_retry as _cwr
        await _cwr(self.session)

        try:
            decision = await llm_provider.generate_structured(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_prompt},
                ],
                SocialDecision,
                # Autonomous joins are her most playful moments — a little
                # extra heat on top of the state-driven temperature.
                temperature=min(
                    settings.decision_temperature_max,
                    self._decision_temperature(emotional_state) + 0.1,
                ),
                max_tokens=2000,
            )
            decision.action = SocialAction.INITIATE
            decision.target_user = last_msg.user_id
            decision.language = language_decision.language
            decision.response_delay = self.social_engine.calculate_response_delay(
                is_direct_mention=False, is_autonomous=True,
                energy=emotional_state.get("energy", 0.5),
            )

            if decision.response:
                # Validate the autonomous response before it's committed.
                validated = await self.validator.validate(
                    decision.response,
                    current_topic=context.state.current_topic,
                    is_reply_to_airi=False,
                    conversation_id=conversation_id,
                    room_avg_words=self._room_avg_words(context),
                )
                if await self.validator.should_skip(validated):
                    logger.info("autonomous_response_blocked", issues=validated["issues"])
                    await self.anti_repetition.record(conversation_id, "response_pattern", decision.response)
                    return None
                decision.response = validated["response"]

                has_snippet = bool(language_decision.allow_persian_snippet and decision.language == "english")
                await self.language_stats.record_response(
                    conversation_id, decision.language or "english", language_decision.reason, has_snippet
                )
                await self.conversation_manager.record_bot_initiation(conversation_id)
                await self.conversation_manager.increment_bot_message(conversation_id)
                self.social_engine.record_message()
                await self.emotional_manager.update_from_event("participation", 0.5)
                await self.anti_repetition.record_response(
                    conversation_id, decision.response, decision.current_topic,
                    target_user_id=last_msg.user_id,
                )
                await self.working_memory.add_airi_action(
                    conversation_id, "initiate", {"text": (decision.response or "")[:200]}
                )
                await self._record_training_sample(
                    conversation_id, "initiate", decision, context, usernames_by_id,
                )

            logger.info(
                "autonomous_participation",
                conversation_id=conversation_id,
                score=round(score, 3),
                attention=round(attention, 3),
                language=decision.language,
                reasoning=decision.reasoning_summary[:100],
            )
            return decision

        except Exception as e:
            logger.error("autonomous_generation_failed", error=str(e))
            return None


    # ------------------------------------------------------------------
    # Memory formation + learning loop runners
    # ------------------------------------------------------------------

    async def run_memory_extraction(self, conversation_id: int):
        from app.memory import MemoryFormation

        context = await self.conversation_manager.get_context(conversation_id)
        if not context or len(context.recent_messages) < 5:
            return []

        formation = MemoryFormation(self.session)
        usernames_by_id = await self._build_usernames_map(context)
        # HER OWN messages are part of the record now: commitments she made
        # ("I'll send it tomorrow") and opinions she stated become memories,
        # so she can follow through and stay consistent. They enter the
        # transcript under her own name.
        messages_data = [
            {
                "user_id": msg.user_id,
                "text": msg.text,
                "username": (
                    self.bot_username if msg.is_from_bot
                    else usernames_by_id.get(msg.user_id, "unknown")
                ),
            }
            for msg in context.recent_messages[-20:]
            if msg.text
        ]

        # Release the read snapshot BEFORE the LLM call. The session has been
        # in a read transaction since get_context(); if that snapshot survived
        # the (30s+) LLM call, the first post-call write would have to UPGRADE
        # read->write — a path SQLite's busy_timeout does NOT cover, which
        # fails instantly with "database is locked" the moment any handler
        # committed meanwhile (then the invalidated session produced the
        # follow-up "connection error").
        from app.database import commit_with_retry
        await commit_with_retry(self.session)

        new_memories = await formation.extract_from_conversation(
            conversation_id, messages_data, usernames_by_id=usernames_by_id,
            persona_name=self.bot_username,
        )
        if new_memories:
            logger.info("memories_formed", conversation_id=conversation_id, count=len(new_memories))
        return new_memories

    async def run_learning_loop(self, conversation_id: int):
        from app.agent.learning import LearningLoop

        context = await self.conversation_manager.get_context(conversation_id)
        if not context or len(context.recent_messages) < 4:
            return None

        usernames_by_id = await self._build_usernames_map(context)
        messages_data = [
            {
                "user_id": msg.user_id,
                "text": msg.text,
                "username": usernames_by_id.get(msg.user_id, "unknown"),
            }
            for msg in context.recent_messages[-25:]
            if msg.text and not msg.is_from_bot
        ]

        # Same snapshot-release as run_memory_extraction: the read transaction
        # must not survive the LLM call (read->write upgrade = "database is
        # locked" the moment any handler committed meanwhile).
        from app.database import commit_with_retry
        await commit_with_retry(self.session)

        loop = LearningLoop(self.session)
        return await loop.learn_from_window(conversation_id, messages_data, usernames_by_id)
