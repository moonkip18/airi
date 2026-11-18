"""Airi's learning loop and self-model.

The LLM's weights are never modified. Instead, experience-based adaptation
happens through persistent state:

* ``SelfModel`` — obsessions, recurring jokes, learned tendencies, topic
  engagement counters. Updated after meaningful conversations.
* Relationship deltas — applied through RelationshipManager.
* Emotional events — applied through EmotionalStateManager.

The learning evaluation is ONE batched LLM call per conversation window
(never per message), and it can conclude "nothing to learn".
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import SelfModel

logger = structlog.get_logger()


class SelfModelManager:
    """Persistent access to Airi's self-development data."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create(self) -> SelfModel:
        stmt = select(SelfModel)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        if model is None:
            model = SelfModel(
                obsessions={}, recurring_jokes=[], learned_tendencies=[],
                topic_engagement={}, stances={},
            )
            self.session.add(model)
            await self.session.flush()
        # Defensive defaults for rows created before all fields existed.
        if model.obsessions is None:
            model.obsessions = {}
        if model.recurring_jokes is None:
            model.recurring_jokes = []
        if model.learned_tendencies is None:
            model.learned_tendencies = []
        if model.topic_engagement is None:
            model.topic_engagement = {}
        if model.stances is None:
            model.stances = {}
        return model

    async def add_topic_engagement(self, topics: Dict[str, float]) -> Dict[str, float]:
        """Accumulate engagement per topic. High engagement over time can
        gradually turn a topic into an obsession — never from one message."""
        model = await self.get_or_create()
        for topic, delta in topics.items():
            key = topic.strip().lower()
            if not key:
                continue
            current = float(model.topic_engagement.get(key, 0.0))
            model.topic_engagement[key] = round(current + float(delta), 3)

        # Promote topics that crossed the obsession threshold.
        promoted = []
        for topic, score in list(model.topic_engagement.items()):
            if score >= settings.obsession_engagement_threshold and topic not in model.obsessions:
                model.obsessions[topic] = round(score, 3)
                promoted.append(topic)
                logger.info("obsession_developed", topic=topic, engagement=round(score, 2))

        # Cap obsessions: keep the strongest.
        if len(model.obsessions) > settings.max_obsessions:
            sorted_obs = sorted(model.obsessions.items(), key=lambda kv: kv[1], reverse=True)
            model.obsessions = dict(sorted_obs[: settings.max_obsessions])

        model.updated_at = datetime.utcnow()
        await self.session.flush()
        return model.topic_engagement

    async def get_obsessions(self) -> Dict[str, float]:
        model = await self.get_or_create()
        return dict(model.obsessions)

    async def decay_engagement(self, factor: float = 0.98):
        """Slowly decay engagement so old topics can fall out of fashion."""
        model = await self.get_or_create()
        for topic in list(model.topic_engagement.keys()):
            model.topic_engagement[topic] = round(model.topic_engagement[topic] * factor, 3)
            if model.topic_engagement[topic] < 0.5 and topic not in model.obsessions:
                del model.topic_engagement[topic]
        model.updated_at = datetime.utcnow()
        await self.session.flush()

    async def add_recurring_joke(self, phrase: str, language: str, origin: str = "") -> bool:
        phrase = (phrase or "").strip()
        if not phrase:
            return False
        model = await self.get_or_create()
        for joke in model.recurring_jokes:
            if str(joke.get("phrase", "")).lower() == phrase.lower():
                joke["times_used"] = int(joke.get("times_used", 0)) + 1
                await self.session.flush()
                return False
        if len(model.recurring_jokes) >= 8:
            model.recurring_jokes.pop(0)
        model.recurring_jokes.append({
            "phrase": phrase,
            "language": language,
            "origin": origin,
            "times_used": 1,
        })
        model.updated_at = datetime.utcnow()
        await self.session.flush()
        logger.info("recurring_joke_learned", phrase=phrase[:40], language=language)
        return True

    async def get_recurring_jokes(self) -> List[Dict[str, Any]]:
        model = await self.get_or_create()
        return list(model.recurring_jokes)

    async def add_learned_tendency(self, tendency: str) -> bool:
        tendency = (tendency or "").strip()
        if not tendency:
            return False
        model = await self.get_or_create()
        if any(t.lower() == tendency.lower() for t in model.learned_tendencies):
            return False
        if len(model.learned_tendencies) >= 12:
            model.learned_tendencies.pop(0)
        model.learned_tendencies.append(tendency)
        model.updated_at = datetime.utcnow()
        await self.session.flush()
        logger.info("learned_tendency_added", tendency=tendency[:80])


class LearningLoop:
    """Post-conversation evaluation: what happened, what did Airi learn?

    ONE structured LLM call per conversation window. It evaluates what
    happened, relationship changes, new obsessions forming, behavioral
    failures, and useful recurring patterns — then applies deltas locally.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.self_model = SelfModelManager(session)

    async def learn_from_window(
        self,
        conversation_id: int,
        messages: List[Dict[str, Any]],
        usernames_by_id: Optional[Dict[int, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        from app.llm import llm_provider
        from typing import Literal as _Literal
        from pydantic import BaseModel as _BaseModel, Field as _Field
        import json

        if not messages:
            return None
        usernames_by_id = usernames_by_id or {}

        class RelationshipDelta(_BaseModel):
            username: str = ""
            interaction_type: _Literal[
                "positive", "negative", "helpful", "interesting", "boring",
                "teasing", "funny", "apology", "disappointment", "jealousy_trigger",
            ] = "positive"
            intensity: float = 0.5

        class EmotionalEvent(_BaseModel):
            event: _Literal[
                "funny", "interesting_topic", "boring", "insult", "compliment",
                "argued", "won_argument", "proven_wrong", "acknowledged_mistake",
                "embarrassed", "ai_reminder", "affection_display", "jealousy_trigger",
            ] = "funny"
            intensity: float = 0.5

        class JokeCandidate(_BaseModel):
            phrase: str = ""
            language: _Literal["english", "persian"] = "english"

        class StanceCandidate(_BaseModel):
            topic: str = ""
            stance: str = ""

        class LearningOutcome(_BaseModel):
            nothing_to_learn: bool = True
            summary: str = ""
            relationship_deltas: List[RelationshipDelta] = _Field(default_factory=list)
            emotional_events: List[EmotionalEvent] = _Field(default_factory=list)
            topic_engagement: Dict[str, float] = _Field(default_factory=dict)
            new_joke: Optional[JokeCandidate] = None
            learned_tendency: str = ""
            stance_update: Optional[StanceCandidate] = None


        transcript_lines = []
        for msg in messages:
            username = msg.get("username") or f"user_{msg.get('user_id', '?')}"
            text = (msg.get("text") or "").strip()
            if text:
                transcript_lines.append(f"{username}: {text}")
        transcript = "\n".join(transcript_lines)
        if not transcript:
            return None

        system = (
            "You are the learning module of an AI persona living in a group chat. "
            "Evaluate the conversation window and report what the persona should "
            "LEARN from it, if anything. Most windows are casual chatter with "
            "nothing durable to learn — then set nothing_to_learn=true. Respond "
            "ONLY with valid JSON matching the schema. Do not invent drama: only "
            "report relationship changes or emotional events actually visible in "
            "the transcript."
        )

        current_jokes = await self.self_model.get_recurring_jokes()
        prompt = (
            f"Conversation window:\n{transcript}\n\n"
            f"username -> user_id map: "
            f"{json.dumps({str(k): v for k, v in usernames_by_id.items()})}\n\n"
            f"Already-known recurring jokes: {json.dumps([j.get('phrase') for j in current_jokes])}\n\n"
            "Report: relationship_deltas (per person), emotional_events, "
            "topic_engagement (topic -> 0.0-2.0 engagement this window showed), "
            "new_joke (only if a genuinely reusable catchphrase emerged, "
            "prefer the persona's own phrasing), learned_tendency (one short "
            "sentence about a stable behavioral pattern, or empty), "
            "stance_update (only when the persona herself committed to a "
            "clear opinion this window: topic = short lowercase key, stance "
            "= her position in a few words, in her voice; else null)."
        )

        try:
            outcome = await llm_provider.generate_structured(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                LearningOutcome,
                temperature=0.3,
                max_tokens=2000,
            )
        except Exception as e:
            logger.error("learning_loop_failed", error=str(e))
            return None

        if outcome.nothing_to_learn:
            logger.debug("learning_loop_nothing_to_learn", conversation_id=conversation_id)
            return None

        # Deltas are applied locally (fast reads + writes, no LLM) under the
        # global write lock — and the commit happens INSIDE the lock: a
        # released-but-uncommitted write transaction keeps SQLite's RESERVED
        # lock and poisons every other writer for a full busy_timeout (30s).
        from app.database.session import db_write_lock, commit_with_retry
        async with db_write_lock:
            applied = await self._apply_outcome(conversation_id, outcome, usernames_by_id)
            await commit_with_retry(self.session)
        logger.info("learning_loop_applied", conversation_id=conversation_id,
                    relationship_deltas=applied["relationship_deltas"], events=applied["events"])
        return applied

    async def _apply_outcome(self, conversation_id: int, outcome, usernames_by_id) -> dict:
        applied: dict = {"summary": outcome.summary, "relationship_deltas": 0, "events": 0}

        # 1. Relationship deltas
        from app.state import RelationshipManager
        rel_manager = RelationshipManager(self.session)
        username_to_id = {str(v).lstrip("@").lower(): k for k, v in usernames_by_id.items()}
        for delta in outcome.relationship_deltas:
            user_id = username_to_id.get(str(delta.username).lstrip("@").lower())
            if user_id is None:
                continue
            await rel_manager.update_from_interaction(
                user_id, delta.interaction_type, max(0.1, min(3.0, delta.intensity))
            )
            applied["relationship_deltas"] += 1

        # 2. Emotional events
        from app.state import EmotionalStateManager
        emo_manager = EmotionalStateManager(self.session)
        for event in outcome.emotional_events:
            await emo_manager.update_from_event(event.event, max(0.1, min(3.0, event.intensity)))
            applied["events"] += 1

        # 3. Topic engagement -> possible obsession development
        if outcome.topic_engagement:
            await self.self_model.add_topic_engagement(outcome.topic_engagement)
            applied["topics"] = outcome.topic_engagement

        # 4. Recurring joke
        if outcome.new_joke and outcome.new_joke.phrase.strip():
            await self.self_model.add_recurring_joke(
                outcome.new_joke.phrase, outcome.new_joke.language,
                origin=f"conversation:{conversation_id}",
            )

        # 5. Learned tendency
        if outcome.learned_tendency.strip():
            await self.self_model.add_learned_tendency(outcome.learned_tendency.strip())

        # 6. Stance update (opinion consistency)
        if (
            outcome.stance_update
            and outcome.stance_update.topic.strip()
            and outcome.stance_update.stance.strip()
        ):
            model = await self.self_model.get_or_create()
            stances = dict(model.stances or {})
            stances[outcome.stance_update.topic.strip().lower()[:60]] = (
                outcome.stance_update.stance.strip()[:200]
            )
            if len(stances) > 10:
                stances = dict(list(stances.items())[-10:])
            model.stances = stances
            model.updated_at = datetime.utcnow()
            await self.session.flush()
            applied["stance"] = outcome.stance_update.topic[:40]

        return applied
