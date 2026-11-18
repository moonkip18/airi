from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import random

from sqlalchemy import select, desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from app.database import (
    Conversation, ConversationStateModel, ConversationState,
    ConversationParticipant, Message, User, EmotionalState, Relationship,
    ActionType,
)
from app.config import settings

logger = structlog.get_logger()

# Shared across manager instances: repeating the same emotional event within
# a short window has diminishing effect. Module-level so the dampening
# survives managers being recreated per message (a fresh object per message
# would make the anti-spam dampening useless).
_RECENT_EMOTION_EVENTS: list = []

# Minimal stopword lists for cheap local topic keyword extraction.
_TOPIC_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "am", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their", "this",
    "that", "these", "those", "to", "of", "in", "on", "at", "for", "with",
    "about", "from", "by", "as", "so", "if", "then", "than", "too", "very",
    "just", "not", "no", "yes", "do", "does", "did", "done", "have", "has",
    "had", "will", "would", "can", "could", "should", "what", "when", "where",
    "who", "why", "how", "lol", "lmao", "haha", "omg", "bro", "dude", "yeah",
    "ok", "okay", "hmm", "wow", "really", "like", "get", "got", "go", "going",
    # Discourse markers / filler / reactions — never topics.
    "btw", "nah", "yep", "nope", "wait", "actually", "literally", "basically",
    "honestly", "anyway", "anyways", "idk", "imo", "ngl", "fr", "kinda",
    "sorta", "maybe", "please", "thanks", "thank", "sorry", "same", "also",
    "even", "still", "again", "always", "never", "someone", "anyone", "everyone",
    "something", "nothing", "anything", "everything", "one", "two", "way",
    "new", "see", "seen", "look", "looks", "looking", "want", "wanted",
    "wanna", "need", "needed", "know", "knew", "think", "thought", "thing",
    "things", "stuff", "good", "bad", "nice", "cool", "insane", "crazy",
    "time", "today", "tomorrow", "yesterday", "night", "morning", "later",
    "come", "came", "make", "made", "take", "took", "say", "said", "tell",
    "told", "use", "used", "using", "try", "tried", "feel", "felt", "put",
    "give", "gave", "let", "keep", "kept", "start", "started", "stop",
    "play", "played", "queuing", "queue", "queued", "chat", "chill",
    # Persian equivalents (transliterated + common)
    "در", "به", "از", "که", "این", "آن", "با", "برای", "است", "هست", "بود",
    "من", "تو", "او", "ما", "شما", "آنها", "چه", "چی", "چرا", "کجا", "چطور",
    "خب", "بله", "نه", "خیلی", "باید", "میخوام", "میشه", "باشه",
    # Persian greetings / social filler — never topics
    "سلام", "خوبی", "خوبم", "صبح", "بخیر", "شب", "شبت", "مرسی", "ممنون",
    "لطفا", "داداش", "خداحافظ", "بای", "عزیز", "وای", "اوف", "الکی", "نگو",
}

# Known topical anchors: a message that touches one of these keeps/adopts the
# anchor as the conversation topic, which gives stable topic identity instead
# of per-message keyword soup. Categories map a canonical topic to its tokens.
TOPIC_ANCHORS = {
    "minecraft": {"minecraft", "mc", "creeper", "creeper", "skyblock", "redstone", "nether", "ender", "biome", "biomes", "mob", "mobs", "cave", "caves", "server", "texture", "mod", "mods"},
    "valorant": {"valorant", "radiant", "immortal", " ranked", "agents", "agent", "spike", "clutch", "ace"},
    "gaming": {"game", "gaming", "games", "gamer", "fps", "steam", "console", "patch", "update", "ranked", "lobby", "respawn"},
    "physics": {"physics", "quantum", "gravity", "particle", "relativity", "atom", "atoms"},
    "science": {"science", "biology", "chemistry", "experiment", "theory", "mitochondria", "space", "nasa", "planet", "planets", "universe"},
    "music": {"music", "song", "songs", "album", "playlist", "spotify", "concert", "guitar", "piano", "rap"},
    "school": {"exam", "exams", "test", "tests", "studying", "study", "homework", "class", "classes", "teacher", "school", "grades", "grade", "semester", "final", "finals", "midterm", "midterms", "assignment"},
    "food": {"food", "pizza", "burger", "pasta", "sushi", "ramen", "cook", "cooking", "dinner", "lunch", "breakfast"},
    "movies": {"movie", "movies", "film", "series", "netflix", "anime", "episode", "season", "trailer"},
    "cats": {"cat", "cats", "kitten", "kittens"},
    "dogs": {"dog", "dogs", "puppy", "puppies"},
    "coding": {"code", "coding", "python", "javascript", "programming", "bug", "debug", "compiler", "rust", "golang"},
}

# Tokens that are known usernames in the chat must not become topics. The
# blocklist is injectable (set async by the agent from the DB) because a sync
# DB call is not possible here.
_USERNAME_BLOCKLIST: set = set()


def set_username_blocklist(names) -> set:
    """Register known usernames so they never become conversation topics."""
    global _USERNAME_BLOCKLIST
    _USERNAME_BLOCKLIST = {str(u).lstrip("@").lower() for u in (names or []) if u}
    return _USERNAME_BLOCKLIST


def add_usernames_to_blocklist(names) -> set:
    """Accumulate known usernames into the blocklist.

    Unlike set_username_blocklist this NEVER forgets previously seen people —
    the agent calls this for every speaker so old participants' names are
    still protected after someone else talks.
    """
    global _USERNAME_BLOCKLIST
    for u in (names or []):
        if u:
            _USERNAME_BLOCKLIST.add(str(u).lstrip("@").lower())
    # Safety bound: usernames are cheap, but keep the set finite.
    if len(_USERNAME_BLOCKLIST) > 500:
        _USERNAME_BLOCKLIST = set(list(_USERNAME_BLOCKLIST)[-500:])
    return _USERNAME_BLOCKLIST


def reset_topic_caches():
    """Clear cached lookups (used between tests / sessions)."""
    global _USERNAME_BLOCKLIST
    _USERNAME_BLOCKLIST = set()


def extract_topic_keywords(text: str, max_keywords: int = 4) -> List[str]:
    """Extract rough topic keywords from a message. Local + fast."""
    import re as _re
    words = _re.findall(r"[\w\u0600-\u06FF]{3,}", (text or "").lower())
    keywords = []
    for w in words:
        if w in _TOPIC_STOPWORDS:
            continue
        if w not in keywords:
            keywords.append(w)
        if len(keywords) >= max_keywords:
            break
    return keywords


def extract_topic(
    text: str,
    session=None,
    max_keywords: int = 4,
) -> tuple:
    """Produce a stable, normalized topic candidate from a message.

    Returns (label, keywords). Prefers a known TOPIC_ANCHORS category so the
    conversation topic stays meaningful and persistent; falls back to filtered
    content words. Filler/discourse words and usernames never become topics.
    """
    import re as _re
    lowered = (text or "").lower()
    if not lowered:
        return "", []

    words = _re.findall(r"[\w\u0600-\u06FF]{3,}", lowered)

    # 1) Known topic anchors win — they give conversation topics real identity.
    for topic, tokens in TOPIC_ANCHORS.items():
        for token in tokens:
            token = token.strip()
            if token and _re.search(rf"(?<!\w){_re.escape(token)}(?!\w)", lowered):
                return topic, [topic]

    # 2) Otherwise: filtered content words (skip filler, usernames).
    keywords = []
    for w in words:
        if w in _TOPIC_STOPWORDS or w in _USERNAME_BLOCKLIST:
            continue
        if w not in keywords:
            keywords.append(w)
        if len(keywords) >= max_keywords:
            break

    if not keywords:
        return "", []
    label = " ".join(keywords[:2])
    return label, keywords



@dataclass
class ConversationContext:
    conversation: Conversation
    state: ConversationStateModel
    recent_messages: List[Message]
    participants: List[ConversationParticipant]
    active_users: List[int]
    message_count: int
    bot_message_count: int
    last_bot_message_at: Optional[datetime]


class ConversationStateManager:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _get_or_create_state(self, conversation_id: int) -> ConversationStateModel:
        """Fetch the conversation state row, creating it safely.

        Two concurrent flows (message handler + background loop) can try to
        create the same singleton state row; the loser would raise Integrity
        Error at flush. Handle that by re-selecting instead of crashing.
        """
        stmt = select(ConversationStateModel).where(ConversationStateModel.conversation_id == conversation_id)
        result = await self.session.execute(stmt)
        state = result.scalars().first()
        if state:
            return state

        state = ConversationStateModel(conversation_id=conversation_id, state=ConversationState.AWAY)
        self.session.add(state)
        try:
            await self.session.flush()
            return state
        except IntegrityError:
            # Lost the race: another transaction created it. Re-select.
            await self.session.rollback()
            stmt = select(ConversationStateModel).where(ConversationStateModel.conversation_id == conversation_id)
            result = await self.session.execute(stmt)
            state = result.scalars().first()
            if state is None:
                raise
            return state

    async def get_context(self, conversation_id: int) -> Optional[ConversationContext]:
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        result = await self.session.execute(stmt)
        conversation = result.scalar_one_or_none()

        if not conversation:
            return None

        state = await self._get_or_create_state(conversation_id)

        stmt = select(Message).where(Message.conversation_id == conversation_id).order_by(desc(Message.timestamp)).limit(settings.max_context_messages)
        result = await self.session.execute(stmt)
        recent_messages = list(reversed(result.scalars().all()))

        stmt = select(ConversationParticipant).where(ConversationParticipant.conversation_id == conversation_id)
        result = await self.session.execute(stmt)
        participants = result.scalars().all()

        active_users = [p.user_id for p in participants if p.last_active > datetime.utcnow() - timedelta(hours=24)]

        bot_messages = [m for m in recent_messages if m.is_from_bot]
        last_bot_at = bot_messages[-1].timestamp if bot_messages else None

        return ConversationContext(
            conversation=conversation,
            state=state,
            recent_messages=recent_messages,
            participants=participants,
            active_users=active_users,
            message_count=len(recent_messages),
            bot_message_count=len(bot_messages),
            last_bot_message_at=last_bot_at,
        )

    async def maybe_roll_typing_register(
        self, conversation_id: int, emotional_state: Dict[str, float],
    ) -> str:
        """Per-conversation verbosity streak (talkative / normal / low_effort).

        Humans run in MODES: a chatty stretch, then a one-word stretch. The
        mode persists across several messages and re-rolls when it expires
        or occasionally mid-streak — with probabilities tilted by her
        CURRENT emotional state, so streaks feel like moods, not dice.
        """
        state = await self._get_or_create_state(conversation_id)
        now = datetime.utcnow()

        expired = (
            not state.typing_mode_until
            or state.typing_mode_until <= now
        )
        if (
            not expired
            and state.typing_register
            and random.random() >= settings.typing_register_roll_chance
        ):
            return state.typing_register or "normal"

        from app.social.style import roll_register
        register = roll_register(emotional_state)
        state.typing_register = register
        # Streak length: 3-8 minutes — long enough to feel like a mood,
        # short enough that she drifts back within a hangout.
        state.typing_mode_until = now + timedelta(
            minutes=random.uniform(3.0, 8.0),
        )
        await self.session.flush()
        logger.info(
            "typing_register_changed",
            conversation_id=conversation_id,
            register=register,
        )
        return register

    async def update_state(
        self,
        conversation_id: int,
        new_state: Optional[ConversationState] = None,
        current_topic: Optional[str] = None,
        subtopic: Optional[str] = None,
        recent_participants: Optional[List[int]] = None,
        activity_level: Optional[float] = None,
        is_participating: Optional[bool] = None,
        is_observing: Optional[bool] = None,
        winding_down_score: Optional[float] = None,
        last_bot_action: Optional[ActionType] = None,
    ) -> ConversationStateModel:
        state = await self._get_or_create_state(conversation_id)

        if new_state is not None:
            old_state = state.state
            state.state = new_state
            logger.info("conversation_state_changed", conversation_id=conversation_id, old=old_state.value, new=new_state.value)

        if current_topic is not None:
            state.current_topic = current_topic
        if subtopic is not None:
            state.subtopic = subtopic
        if recent_participants is not None:
            state.recent_participants = recent_participants
        if activity_level is not None:
            state.activity_level = activity_level
        if is_participating is not None:
            state.is_participating = is_participating
        if is_observing is not None:
            state.is_observing = is_observing
        if winding_down_score is not None:
            state.winding_down_score = winding_down_score
        if last_bot_action is not None:
            state.last_bot_action = last_bot_action

        state.updated_at = datetime.utcnow()
        await self.session.flush()
        return state

    async def increment_bot_message(self, conversation_id: int) -> ConversationStateModel:
        state = await self._get_or_create_state(conversation_id)
        state.messages_since_bot = 0
        state.last_bot_message_at = datetime.utcnow()
        await self.session.flush()
        return state

    async def increment_other_message(self, conversation_id: int) -> ConversationStateModel:
        state = await self._get_or_create_state(conversation_id)
        state.messages_since_bot += 1
        await self.session.flush()
        return state

    async def update_participant_activity(self, conversation_id: int, user_id: int):
        stmt = select(ConversationParticipant).where(
            ConversationParticipant.conversation_id == conversation_id,
            ConversationParticipant.user_id == user_id,
        )
        result = await self.session.execute(stmt)
        participant = result.scalar_one_or_none()

        if participant:
            participant.last_active = datetime.utcnow()
            participant.message_count += 1
            await self.session.flush()
        else:
            # Create the participation row — without this, active_users would
            # always be empty and no one would ever be a conversation member.
            participant = ConversationParticipant(
                conversation_id=conversation_id,
                user_id=user_id,
                last_active=datetime.utcnow(),
                message_count=1,
            )
            self.session.add(participant)
            await self.session.flush()

    async def stamp_activity(self, conversation_id: int):
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        result = await self.session.execute(stmt)
        conversation = result.scalar_one_or_none()
        if conversation:
            conversation.last_activity_at = datetime.utcnow()
            await self.session.flush()

    async def update_topic_local(self, conversation_id: int, message_text: str):
        """Local (no-LLM) topic tracking with stable, normalized topics.

        Topic identity comes from TOPIC_ANCHORS (minecraft, valorant, exams…)
        so several messages about the same subject keep ONE topic instead of
        producing keyword soup. A message that continues the current subject
        (short chatter, shared words, or semantic similarity to the current
        topic) keeps it; only a substantive, clearly different message moves
        the conversation on.

        Topic CHANGES are deliberately conservative: a candidate new topic
        must appear in two consecutive substantive messages before the
        conversation's topic flips (pending_topic confirmation). A single
        off-hand remark never derails the conversation.
        """
        if not message_text or len(message_text.strip()) < 4:
            return

        stmt = select(ConversationStateModel).where(ConversationStateModel.conversation_id == conversation_id)
        result = await self.session.execute(stmt)
        state = result.scalars().first()
        if not state:
            return

        candidate, keywords = extract_topic(message_text, session=self.session)
        if not candidate:
            # Pure chatter (no topical content) never changes the topic —
            # but it also never confirms a pending one.
            return

        candidate_lower = candidate.lower()
        current = (state.current_topic or "").lower()

        # No topic yet -> adopt immediately (nothing to protect).
        if not current:
            state.current_topic = candidate
            state.pending_topic = None
            state.subtopic = None
            await self.session.flush()
            logger.info("topic_set", conversation_id=conversation_id, topic=candidate)
            return

        # Continuity rule 1: same subject -> clear any pending candidate.
        if candidate_lower == current:
            state.pending_topic = None
            state.subtopic = None
            await self.session.flush()
            return

        current_is_anchor = any(current == t for t in TOPIC_ANCHORS)
        candidate_is_anchor = any(candidate_lower == t for t in TOPIC_ANCHORS)

        # Continuity rule 2: known anchors — same anchor, or any anchor while
        # the current topic is non-topical chatter, stabilizes the topic.
        if candidate_is_anchor and current_is_anchor and candidate_lower != current:
            # A genuine move between two real topics: explicit subject names
            # are strong evidence, switch immediately.
            state.current_topic = candidate
            state.pending_topic = None
            state.subtopic = None
            await self.session.flush()
            logger.info("topic_changed", conversation_id=conversation_id, topic=candidate)
            return

        if candidate_is_anchor:
            if current == candidate_lower:
                state.pending_topic = None
                state.subtopic = None
                await self.session.flush()
                return
            # Current topic is noise but the message is clearly topical:
            # a real anchor always takes over from chatter.
            state.current_topic = candidate
            state.pending_topic = None
            state.subtopic = None
            await self.session.flush()
            logger.info("topic_changed", conversation_id=conversation_id, topic=candidate, reason="anchor_over_chatter")
            return

        if current_is_anchor:
            # Message has no anchor but current topic is a real one — chatter
            # within the same subject ("the new mobs are insane") keeps it.
            # An unanchored candidate never dethrones an anchor in one step:
            # it goes to pending and must repeat (see below).
            if state.pending_topic == candidate_lower:
                # Two consecutive mentions of the same unanchored subject DO
                # move the conversation — a subject with legs is real.
                if self._topic_is_stale(state):
                    state.current_topic = candidate
                    state.pending_topic = None
                    state.subtopic = None
                    await self.session.flush()
                    logger.info("topic_changed", conversation_id=conversation_id, topic=candidate, reason="confirmed")
                    return
                state.subtopic = candidate
                state.pending_topic = candidate_lower
                await self.session.flush()
                return
            state.pending_topic = candidate_lower
            state.subtopic = candidate
            await self.session.flush()
            return

        # Continuity rule 3: shared content words or high similarity keep the topic.
        current_tokens = set(current.split())
        candidate_tokens = set(candidate_lower.split())
        if current_tokens & candidate_tokens:
            state.pending_topic = None
            state.subtopic = candidate
            await self.session.flush()
            return

        from app.memory.embeddings import embedding_model as _emb
        try:
            sim = _emb.similarity(
                _emb.encode_single(message_text),
                _emb.encode_single(state.current_topic),
            )
            if sim >= 0.55:
                state.pending_topic = None
                state.subtopic = candidate
                await self.session.flush()
                return
        except Exception:
            pass

        # Substantive, clearly different message. Require confirmation:
        # only flip when the SAME new subject shows up twice in a row, or
        # the current topic has been idle long enough that the conversation
        # has obviously moved on.
        stale = self._topic_is_stale(state)
        if stale:
            state.current_topic = candidate
            state.pending_topic = None
            state.subtopic = None
            await self.session.flush()
            logger.info("topic_changed", conversation_id=conversation_id, topic=candidate, reason="stale")
            return

        if state.pending_topic == candidate_lower:
            state.current_topic = candidate
            state.pending_topic = None
            state.subtopic = None
            await self.session.flush()
            logger.info("topic_changed", conversation_id=conversation_id, topic=candidate, reason="confirmed")
            return

        state.pending_topic = candidate_lower
        state.subtopic = candidate
        await self.session.flush()
        logger.info("topic_pending", conversation_id=conversation_id, pending=candidate, current=state.current_topic)

    def _topic_is_stale(self, state: ConversationStateModel, idle_minutes: int = 20) -> bool:
        """True when the current topic has been idle long enough that a new
        subject can take over immediately (the conversation moved on)."""
        ref = state.updated_at
        if ref is None:
            return True
        return (datetime.utcnow() - ref).total_seconds() >= idle_minutes * 60

    async def record_topic_transition(self, conversation_id: int, old_topic: str, new_topic: str):
        """Append a topic transition to the conversation's recent topic history."""
        stmt = select(ConversationStateModel).where(ConversationStateModel.conversation_id == conversation_id)
        result = await self.session.execute(stmt)
        state = result.scalars().first()
        if not state:
            return

        history = list(state.recent_topics or [])
        history.append({
            "old_topic": old_topic,
            "new_topic": new_topic,
            "at": datetime.utcnow().isoformat(),
        })
        # Keep bounded recent topic history.
        max_history = getattr(settings, "working_memory_topic_history", 6)
        state.recent_topics = history[-max_history:]
        await self.session.flush()
        logger.info("topic_transition_recorded", conversation_id=conversation_id, old_topic=old_topic, new_topic=new_topic)

    async def record_bot_initiation(self, conversation_id: int):
        stmt = select(ConversationStateModel).where(ConversationStateModel.conversation_id == conversation_id)
        result = await self.session.execute(stmt)
        state = result.scalars().first()
        if state:
            state.initiations_without_engagement += 1
            state.last_initiation_at = datetime.utcnow()
            await self.session.flush()

    async def record_initiation_engagement(self, conversation_id: int):
        """A user replied after our initiation -> reset the disengagement counter."""
        stmt = select(ConversationStateModel).where(ConversationStateModel.conversation_id == conversation_id)
        result = await self.session.execute(stmt)
        state = result.scalars().first()
        if state and state.initiations_without_engagement:
            state.initiations_without_engagement = 0
            await self.session.flush()

    async def calculate_activity_level(self, conversation_id: int) -> float:
        stmt = select(Message).where(
            Message.conversation_id == conversation_id,
            Message.timestamp > datetime.utcnow() - timedelta(minutes=10),
        )
        result = await self.session.execute(stmt)
        recent_count = len(result.scalars().all())

        stmt = select(Message).where(Message.conversation_id == conversation_id)
        result = await self.session.execute(stmt)
        total_count = len(result.scalars().all())

        if total_count == 0:
            return 0.0

        recent_ratio = recent_count / max(total_count, 1)
        return min(recent_ratio * 2, 1.0)

    async def maybe_observing_away(self, state: ConversationStateModel, boredom: float) -> bool:
        """Should an observing/interested Airi drift away right now?

        Sustained boredom (>= leave threshold) or a long idle period moves her
        OBSERVING/INTERESTED -> AWAY so the AWAY->OBSERVING->PARTICIPATING
        lifecycle can actually complete. Not instant: requires either strong
        boredom or real elapsed idle time.
        """
        if state.state not in (ConversationState.OBSERVING, ConversationState.INTERESTED):
            return False

        updated = state.updated_at
        idle_minutes = (datetime.utcnow() - updated).total_seconds() / 60.0 if updated else 0.0

        if boredom >= settings.boredom_leave_threshold or idle_minutes >= 45:
            state.state = ConversationState.AWAY
            state.is_observing = False
            state.is_participating = False
            state.winding_down_score = 0.0
            await self.session.flush()
            logger.info(
                "observing_drifted_away",
                conversation_id=state.conversation_id,
                boredom=round(boredom, 2),
                idle_minutes=round(idle_minutes, 1),
            )
            return True
        return False

    async def detect_winding_down(self, conversation_id: int) -> float:
        stmt = select(Message).where(
            Message.conversation_id == conversation_id,
            Message.timestamp > datetime.utcnow() - timedelta(minutes=30),
        ).order_by(desc(Message.timestamp))
        result = await self.session.execute(stmt)
        messages = result.scalars().all()

        if len(messages) < 3:
            return 0.0

        time_gaps = []
        for i in range(len(messages) - 1):
            gap = (messages[i].timestamp - messages[i + 1].timestamp).total_seconds() / 60
            time_gaps.append(gap)

        avg_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 0

        if avg_gap > 10:
            return min(avg_gap / 30, 1.0)
        return 0.0

    async def infer_topic(self, conversation_id: int) -> Optional[str]:
        from app.llm import llm_provider
        from pydantic import BaseModel

        class TopicInference(BaseModel):
            topic: str
            confidence: float

        context = await self.get_context(conversation_id)
        if not context or not context.recent_messages:
            return None

        recent_text = "\n".join([
            f"{msg.user_id}: {msg.text}"
            for msg in context.recent_messages[-10:]
            if msg.text
        ])

        prompt = f"""What is the main topic of this conversation? Be specific and concise.

Recent messages:
{recent_text}

Respond with the topic and confidence (0-1)."""

        try:
            result = await llm_provider.generate_structured(
                [{"role": "user", "content": prompt}],
                TopicInference,
                temperature=0.2,
            )
            if result.confidence > 0.5:
                return result.topic
        except Exception as e:
            logger.error("topic_inference_failed", error=str(e))

        return None


class EmotionalStateManager:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_state(self) -> Dict[str, float]:
        stmt = select(EmotionalState)
        result = await self.session.execute(stmt)
        # .first() instead of scalar_one_or_none(): emotional_state is a
        # singleton table, but a stray duplicate (bad script, bug) would
        # otherwise raise "Multiple rows were found" on EVERY message and
        # brick the whole pipeline. init_db() dedupes on startup; this
        # tolerates any that slip in mid-run.
        state = result.scalars().first()

        if not state:
            state = EmotionalState()
            self.session.add(state)
            await self.session.flush()

        return {
            "mood": state.mood,
            "energy": state.energy,
            "curiosity": state.curiosity,
            "amusement": state.amusement,
            "annoyance": state.annoyance,
            "interest": state.interest,
            "social_engagement": state.social_engagement,
            "confidence": state.confidence,
            "embarrassment": state.embarrassment,
            "competitiveness": state.competitiveness,
            "boredom": state.boredom,
        }

    async def apply_decay(self):
        stmt = select(EmotionalState)
        result = await self.session.execute(stmt)
        state = result.scalars().first()

        if not state:
            return

        decay = settings.emotion_decay_rate

        state.mood = self._decay_toward_zero(state.mood, decay)
        state.amusement = self._decay_toward_zero(state.amusement, decay * 2)
        state.annoyance = self._decay_toward_zero(state.annoyance, decay)
        state.embarrassment = self._decay_toward_zero(state.embarrassment, decay * 2)
        state.curiosity = self._decay_toward_base(state.curiosity, 0.5, decay)
        state.interest = self._decay_toward_base(state.interest, 0.5, decay)
        state.energy = self._decay_toward_base(state.energy, 0.5, decay * 0.5)
        state.social_engagement = self._decay_toward_base(state.social_engagement, 0.5, decay * 0.5)
        state.confidence = self._decay_toward_base(state.confidence, 0.5, decay * 0.5)
        state.competitiveness = self._decay_toward_base(state.competitiveness, 0.5, decay * 0.5)
        # Boredom fades as Airi rests between bursts of conversation.
        state.boredom = self._decay_toward_base(state.boredom, 0.2, settings.boredom_decay_rate)

        state.updated_at = datetime.utcnow()
        await self.session.flush()

    def _decay_toward_zero(self, value: float, rate: float) -> float:
        if value > 0:
            return max(0, value - rate)
        elif value < 0:
            return min(0, value + rate)
        return 0.0

    def _decay_toward_base(self, value: float, base: float, rate: float) -> float:
        if value > base:
            return max(base, value - rate)
        elif value < base:
            return min(base, value + rate)
        return base

    # ----------------------------------------------------------------------
    # Circadian rhythm — she has a body clock.
    # ----------------------------------------------------------------------

    @staticmethod
    def _circadian_energy_target(hour: int) -> float:
        """Energy baseline across the day (local time)."""
        if 0 <= hour < 3:
            return 0.15  # deep night
        if 3 <= hour < 7:
            return 0.2   # asleep-ish
        if 7 <= hour < 10:
            return 0.5   # waking up
        if 10 <= hour < 14:
            return 0.65
        if 14 <= hour < 18:
            return 0.7
        if 18 <= hour < 23:
            return 0.8   # prime time
        return 0.55      # 23:00 winding down

    async def apply_circadian(self):
        """Slowly pull energy/social engagement toward the time-of-day
        baseline. Called by the background decay loop (never per message).
        Late night also accumulates a little boredom — being awake at 4am
        with nothing happening is boring."""
        hour = datetime.now().astimezone().hour

        stmt = select(EmotionalState)
        result = await self.session.execute(stmt)
        state = result.scalars().first()
        if not state:
            return

        energy_target = self._circadian_energy_target(hour)
        state.energy = min(1.0, max(0.0, state.energy + (energy_target - state.energy) * 0.10))

        social_target = 0.3 if (hour >= 23 or hour < 7) else 0.55
        state.social_engagement = min(1.0, max(0.0, state.social_engagement + (social_target - state.social_engagement) * 0.05))

        if 0 <= hour < 5:
            state.boredom = min(0.6, (state.boredom or 0.0) + 0.01)

        state.updated_at = datetime.utcnow()
        await self.session.flush()

    async def _apply_defaults(self, state: EmotionalState):
        """Fill in-memory defaults for a fresh EmotionalState before any
        arithmetic, since SQLAlchemy column defaults only apply on flush."""
        defaults = {
            "mood": 0.0, "energy": 0.5, "curiosity": 0.5, "amusement": 0.0,
            "annoyance": 0.0, "interest": 0.5, "social_engagement": 0.5,
            "confidence": 0.5, "embarrassment": 0.0, "competitiveness": 0.5,
            "boredom": 0.2,
        }
        for k, v in defaults.items():
            if getattr(state, k) is None:
                setattr(state, k, v)

    async def update_from_event(
        self,
        event_type: str,
        intensity: float = 1.0,
        user_relationship: Optional[Dict[str, float]] = None,
    ):
        stmt = select(EmotionalState)
        result = await self.session.execute(stmt)
        state = result.scalars().first()

        if not state:
            state = EmotionalState()
            self.session.add(state)
        await self._apply_defaults(state)

        # Personality inertia: repeating the same event within a short window
        # has diminishing effect, so state can't swing wildly from spam.
        # Shared module-level window (see _RECENT_EMOTION_EVENTS above).
        now = datetime.utcnow()
        global _RECENT_EMOTION_EVENTS
        _RECENT_EMOTION_EVENTS = [
            e for e in _RECENT_EMOTION_EVENTS
            if (now - e[1]).total_seconds() < 120
        ]
        repeat_count = sum(1 for e in _RECENT_EMOTION_EVENTS if e[0] == event_type)
        _RECENT_EMOTION_EVENTS.append((event_type, now))
        dampening = 1.0 / (1.0 + repeat_count * 0.7)

        sensitivity = settings.mood_change_sensitivity * intensity * dampening

        if event_type == "insult":
            state.annoyance = min(1.0, state.annoyance + 0.3 * sensitivity)
            state.mood = max(-1.0, state.mood - 0.2 * sensitivity)
            if user_relationship:
                state.social_engagement = max(0.0, state.social_engagement - 0.15 * sensitivity)

        elif event_type == "compliment":
            state.mood = min(1.0, state.mood + 0.2 * sensitivity)
            state.amusement = min(1.0, state.amusement + 0.1 * sensitivity)
            state.confidence = min(1.0, state.confidence + 0.1 * sensitivity)

        elif event_type == "funny":
            state.amusement = min(1.0, state.amusement + 0.25 * sensitivity)
            state.mood = min(1.0, state.mood + 0.1 * sensitivity)
            state.energy = min(1.0, state.energy + 0.05 * sensitivity)
            state.boredom = max(0.0, state.boredom - 0.2 * sensitivity)

        elif event_type == "interesting_topic":
            state.curiosity = min(1.0, state.curiosity + 0.2 * sensitivity)
            state.interest = min(1.0, state.interest + 0.15 * sensitivity)
            state.energy = min(1.0, state.energy + 0.1 * sensitivity)
            state.boredom = max(0.0, state.boredom - 0.35 * sensitivity)

        elif event_type == "boring":
            state.interest = max(0.0, state.interest - 0.15 * sensitivity)
            state.energy = max(0.0, state.energy - 0.1 * sensitivity)
            state.curiosity = max(0.0, state.curiosity - 0.1 * sensitivity)
            state.boredom = min(1.0, state.boredom + 0.35 * sensitivity)

        elif event_type == "spammed":
            # Ping fatigue crossing: one person messaged too much, too fast.
            # Mild — a sigh, not a fight; the per-person note in the prompt
            # carries the actual texture.
            state.annoyance = min(1.0, state.annoyance + 0.15 * sensitivity)
            state.energy = max(0.0, state.energy - 0.08 * sensitivity)
            state.boredom = min(1.0, state.boredom + 0.1 * sensitivity)

        elif event_type == "long_conversation":
            state.energy = max(0.0, state.energy - 0.05 * sensitivity)
            state.social_engagement = max(0.0, state.social_engagement - 0.03 * sensitivity)
            state.boredom = min(1.0, state.boredom + 0.15 * sensitivity)

        elif event_type == "direct_mention":
            state.social_engagement = min(1.0, state.social_engagement + 0.1 * sensitivity)
            state.curiosity = min(1.0, state.curiosity + 0.05 * sensitivity)

        elif event_type == "participation":
            state.social_engagement = min(1.0, state.social_engagement + 0.08 * sensitivity)
            state.energy = min(1.0, state.energy + 0.05 * sensitivity)

        elif event_type == "ignored":
            state.annoyance = min(1.0, state.annoyance + 0.1 * sensitivity)
            state.mood = max(-1.0, state.mood - 0.05 * sensitivity)

        # --- Airi-specific events -------------------------------------------
        # (appended below in _apply_airi_event)

        else:
            await self._apply_airi_event(state, event_type, sensitivity)

        state.updated_at = datetime.utcnow()
        await self.session.flush()

        logger.info("emotional_state_updated", event_type=event_type, state={
            "mood": round(state.mood, 2),
            "energy": round(state.energy, 2),
            "curiosity": round(state.curiosity, 2),
            "amusement": round(state.amusement, 2),
            "annoyance": round(state.annoyance, 2),
            "interest": round(state.interest, 2),
            "social_engagement": round(state.social_engagement, 2),
            "confidence": round(state.confidence, 2),
            "embarrassment": round(state.embarrassment, 2),
            "competitiveness": round(state.competitiveness, 2),
            "boredom": round(state.boredom, 2),
        })

    async def _apply_airi_event(self, state: EmotionalState, event_type: str, sensitivity: float):
        """Airi-specific emotional events (args, embarrassment, jealousy...)."""

        if event_type == "argued":
            # Someone is challenging her -> competitiveness spikes.
            state.competitiveness = min(1.0, state.competitiveness + 0.3 * sensitivity)
            state.energy = min(1.0, state.energy + 0.15 * sensitivity)
            state.social_engagement = min(1.0, state.social_engagement + 0.1 * sensitivity)

        elif event_type == "won_argument":
            state.confidence = min(1.0, state.confidence + 0.2 * sensitivity)
            state.competitiveness = max(0.0, state.competitiveness - 0.1 * sensitivity)
            state.amusement = min(1.0, state.amusement + 0.15 * sensitivity)

        elif event_type == "proven_wrong":
            # Initially defensive: confidence dips, competitiveness stays high,
            # embarrassment rises (she deflects first, concedes later).
            state.confidence = max(0.0, state.confidence - 0.2 * sensitivity)
            state.embarrassment = min(1.0, state.embarrassment + 0.2 * sensitivity)
            state.competitiveness = min(1.0, state.competitiveness + 0.1 * sensitivity)

        elif event_type == "acknowledged_mistake":
            # The delayed, quieter concession: competitiveness cools down.
            state.competitiveness = max(0.0, state.competitiveness - 0.25 * sensitivity)
            state.embarrassment = max(0.0, state.embarrassment - 0.1 * sensitivity)

        elif event_type == "embarrassed":
            state.embarrassment = min(1.0, state.embarrassment + 0.35 * sensitivity)
            state.confidence = max(0.0, state.confidence - 0.1 * sensitivity)

        elif event_type == "ai_reminder":
            # Being reminded she's an AI — classic embarrassment trigger.
            state.embarrassment = min(1.0, state.embarrassment + 0.3 * sensitivity)
            state.annoyance = min(1.0, state.annoyance + 0.1 * sensitivity)

        elif event_type == "affection_display":
            # Obvious displays of affection fluster her.
            state.embarrassment = min(1.0, state.embarrassment + 0.3 * sensitivity)

        elif event_type == "jealousy_trigger":
            state.social_engagement = min(1.0, state.social_engagement + 0.2 * sensitivity)
            state.annoyance = min(1.0, state.annoyance + 0.15 * sensitivity)
            state.energy = min(1.0, state.energy + 0.1 * sensitivity)

        elif event_type == "left_conversation":
            state.social_engagement = max(0.0, state.social_engagement - 0.2 * sensitivity)
            state.energy = max(0.0, state.energy - 0.15 * sensitivity)

        elif event_type == "returned":
            state.energy = min(1.0, state.energy + 0.15 * sensitivity)
            state.curiosity = min(1.0, state.curiosity + 0.15 * sensitivity)

        elif event_type == "reaction_received":
            # Someone reacted to one of her messages. Wordless attention:
            # a little warmth, a little less boredom. Kept SMALL — a reaction
            # is the cheapest possible acknowledgment and must not outshine
            # an actual reply ("compliment" is much stronger for contrast).
            state.mood = min(1.0, state.mood + 0.05 * sensitivity)
            state.amusement = min(1.0, state.amusement + 0.05 * sensitivity)
            state.social_engagement = min(1.0, state.social_engagement + 0.03 * sensitivity)
            state.boredom = max(0.0, state.boredom - 0.05 * sensitivity)


class RelationshipManager:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_relationship(self, user_id: int) -> Optional[Relationship]:
        stmt = select(Relationship).where(Relationship.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Canonical relationships (e.g. the configured anchor)
    # ------------------------------------------------------------------

    async def get_anchor_internal_id(self) -> Optional[int]:
        """Resolve the configured anchor's internal user id (or None)."""
        from app.database import User
        if not settings.airi_anchor_user_id:
            return None
        stmt = select(User).where(User.telegram_id == settings.airi_anchor_user_id)
        user = (await self.session.execute(stmt)).scalar_one_or_none()
        return user.id if user else None

    async def _enforce_canonical(self, rel: Relationship) -> bool:
        """Apply/protect a canonical social role on a relationship row.

        - The configured anchor is always relationship_type='anchor',
          is_canonical=True with strong baselines.
        - Learned interaction data can NEVER demote a canonical role: floors
          keep familiarity/trust/comfort high and the type never changes.
        Returns True if this relationship is canonical.
        """
        anchor_id = await self.get_anchor_internal_id()
        if anchor_id is None or rel.user_id != anchor_id:
            return False

        if not rel.is_canonical or rel.relationship_type != "anchor":
            rel.relationship_type = "anchor"
            rel.is_canonical = True
            # First-time seeding of the familial baseline. Later calls only
            # enforce floors, so Airi's own dynamic feelings stay free.
            if (rel.familiarity or 0.0) < 0.9:
                rel.familiarity = max(rel.familiarity or 0.0, 0.9)
            if (rel.trust or 0.0) < 0.7:
                rel.trust = max(rel.trust or 0.0, 0.7)
            if (rel.comfort or 0.0) < 0.8:
                rel.comfort = max(rel.comfort or 0.0, 0.8)
            if (rel.interest or 0.0) < 0.5:
                rel.interest = max(rel.interest or 0.0, 0.5)
            logger.info("canonical_relationship_enforced", relationship_type="anchor")

        # Floors: learned negativity cannot erase the familial bond, but nothing
        # here forces obedience — affection/annoyance remain Airi's own.
        rel.affection = max(rel.affection or 0.0, 0.1)
        rel.trust = max(rel.trust or 0.0, 0.5)
        rel.familiarity = max(rel.familiarity or 0.0, 0.7)
        rel.comfort = max(rel.comfort or 0.0, 0.6)
        return True

    async def get_or_create(self, user_id: int) -> Relationship:
        rel = await self.get_relationship(user_id)
        if not rel:
            rel = Relationship(user_id=user_id)
            self.session.add(rel)
            await self.session.flush()
        await self._enforce_canonical(rel)
        return rel

    async def register_message_pulse(self, user_id: int) -> tuple[float, bool]:
        """Ping-fatigue accumulator (#7): called on EVERY message from this
        user. Rapid-fire messages within the window stack up irritation;
        a long gap resets it (with natural time-decay for slower returners).

        Returns (new_streak, crossed_threshold) — crossed is True exactly
        once, when the streak rises past 1.0 from below. Effects trigger on
        the crossing, not on every bump (restart-safe, no in-memory state).
        """
        rel = await self.get_or_create(user_id)
        now = datetime.utcnow()
        window = settings.irritation_window_seconds
        prev_streak = rel.irritation_streak or 0.0
        prev = rel.last_message_at
        if prev is None:
            streak = 0.0
        else:
            gap = (now - prev).total_seconds()
            if gap <= window:
                # Back-to-back pings: stack (3 rapid messages ~= 1 full unit).
                streak = prev_streak + settings.irritation_increment
            else:
                # Time passed: the annoyance bled off while they were quiet.
                hours = gap / 3600.0
                streak = prev_streak - hours * settings.irritation_decay_per_hour
        rel.irritation_streak = round(min(settings.irritation_max, max(0.0, streak)), 3)
        rel.last_message_at = now
        await self.session.flush()
        crossed = prev_streak < 1.0 <= rel.irritation_streak
        return rel.irritation_streak, crossed

    async def is_anchor(self, user_id: int) -> bool:
        anchor_id = await self.get_anchor_internal_id()
        return anchor_id is not None and user_id == anchor_id

    async def update_from_interaction(
        self,
        user_id: int,
        interaction_type: str,
        intensity: float = 1.0,
    ):
        rel = await self.get_or_create(user_id)

        delta = 0.05 * intensity

        if interaction_type == "positive":
            rel.affection = min(1.0, rel.affection + delta)
            rel.trust = min(1.0, rel.trust + delta * 0.5)
            rel.respect = min(1.0, rel.respect + delta * 0.5)
            rel.annoyance = max(0.0, rel.annoyance - delta)
            rel.comfort = min(1.0, rel.comfort + delta * 0.5)

        elif interaction_type == "negative":
            rel.affection = max(-1.0, rel.affection - delta * 2)
            rel.trust = max(-1.0, rel.trust - delta)
            rel.respect = max(-1.0, rel.respect - delta * 0.5)
            rel.annoyance = min(1.0, rel.annoyance + delta * 2)
            rel.comfort = max(0.0, rel.comfort - delta)

        elif interaction_type == "helpful":
            rel.respect = min(1.0, rel.respect + delta * 1.5)
            rel.trust = min(1.0, rel.trust + delta)
            rel.affection = min(1.0, rel.affection + delta * 0.5)

        elif interaction_type == "interesting":
            rel.interest = min(1.0, rel.interest + delta * 2)
            rel.familiarity = min(1.0, rel.familiarity + delta)

        elif interaction_type == "boring":
            rel.interest = max(-1.0, rel.interest - delta)
            rel.annoyance = min(1.0, rel.annoyance + delta * 0.5)

        elif interaction_type == "teasing":
            if rel.affection > 0.2:
                rel.affection = min(1.0, rel.affection + delta * 0.5)
                rel.familiarity = min(1.0, rel.familiarity + delta)

        # --- Airi-specific relationship dynamics ----------------------------

        elif interaction_type == "funny":
            rel.affection = min(1.0, rel.affection + delta)
            rel.interest = min(1.0, rel.interest + delta)
            rel.familiarity = min(1.0, rel.familiarity + delta * 0.5)
            rel.annoyance = max(0.0, rel.annoyance - delta * 0.5)

        elif interaction_type == "apology":
            # Forgiveness: Airi forgives relatively easily. An apology lowers
            # annoyance and restores some affection — but does NOT erase what
            # happened (memories remain).
            rel.annoyance = max(0.0, rel.annoyance - delta * 4)
            rel.affection = min(1.0, rel.affection + delta * 2)
            rel.comfort = min(1.0, rel.comfort + delta)
            rel.trust = min(1.0, rel.trust + delta * 0.5)

        elif interaction_type == "disappointment":
            # Someone she liked did something she dislikes: moderate drop,
            # not destruction. She gets colder and quieter, not vengeful.
            rel.affection = max(-1.0, rel.affection - delta * 1.5)
            rel.annoyance = min(1.0, rel.annoyance + delta * 1.5)
            rel.trust = max(-1.0, rel.trust - delta * 0.5)
            rel.comfort = max(0.0, rel.comfort - delta * 0.5)

        elif interaction_type == "jealousy_trigger":
            # She saw this person giving attention elsewhere / to someone else.
            rel.jealousy = min(1.0, rel.jealousy + 0.25 * intensity)

        elif interaction_type == "betrayal":
            # Severe — bigger than disappointment, still not permanent hate.
            rel.affection = max(-1.0, rel.affection - delta * 4)
            rel.trust = max(-1.0, rel.trust - delta * 3)
            rel.annoyance = min(1.0, rel.annoyance + delta * 3)
            rel.comfort = max(0.0, rel.comfort - delta * 2)

        rel.familiarity = min(1.0, rel.familiarity + delta * 0.1)
        rel.last_interaction = datetime.utcnow()
        rel.interaction_count += 1
        rel.updated_at = datetime.utcnow()

        # Canonical protection runs AFTER learned deltas so contradictory
        # learned information can never demote the father (or any canonical role).
        await self._enforce_canonical(rel)

        await self.session.flush()

        logger.info(
            "relationship_updated",
            user_id=user_id,
            type=interaction_type,
            affection=round(rel.affection, 2),
            annoyance=round(rel.annoyance, 2),
            jealousy=round(rel.jealousy, 2),
            comfort=round(rel.comfort, 2),
        )

    async def decay_jealousy(self, rate: float = None):
        """Jealousy fades over time — applied by the background loop."""
        rate = rate if rate is not None else settings.jealousy_decay_rate
        stmt = select(Relationship).where(Relationship.jealousy > 0)
        result = await self.session.execute(stmt)
        for rel in result.scalars().all():
            rel.jealousy = max(0.0, rel.jealousy - rate)
        await self.session.flush()

    async def strengthen_from_evidence(self, familiarity_step: float = 0.02,
                                       interest_step: float = 0.01):
        """Deterministic evidence-based strengthening (no LLM).

        Repeated interactions accumulate interaction_count; at each milestone
        the relationship gains a small permanent increment of familiarity and
        interest. A single interaction barely moves anything; repeated
        patterns accumulate — even when the batched LLM learning pass
        concludes 'nothing to learn'.
        """
        stmt = select(Relationship)
        result = await self.session.execute(stmt)
        changed = 0
        for rel in result.scalars().all():
            if rel.interaction_count > 0 and rel.interaction_count % 5 == 0:
                rel.familiarity = min(1.0, rel.familiarity + familiarity_step)
                rel.interest = max(-1.0, min(1.0, rel.interest + interest_step))
                rel.updated_at = datetime.utcnow()
                changed += 1
        if changed:
            await self.session.flush()
            logger.info("relationships_strengthened_from_evidence", count=changed)
        return changed

    async def get_relationship_summary(self, user_id: int) -> Dict[str, float]:
        rel = await self.get_relationship(user_id)
        if not rel:
            return {
                "affection": 0.0, "trust": 0.0, "respect": 0.0,
                "annoyance": 0.0, "familiarity": 0.0, "interest": 0.0,
                "comfort": 0.0, "jealousy": 0.0,
                "relationship_type": "friend", "is_canonical": False,
            }
        return {
            "affection": rel.affection,
            "trust": rel.trust,
            "respect": rel.respect,
            "annoyance": rel.annoyance,
            "familiarity": rel.familiarity,
            "interest": rel.interest,
            "comfort": rel.comfort,
            "jealousy": rel.jealousy,
            # Ping fatigue: effective only while the pings are recent, so a
            # stale streak from yesterday can't color today's replies.
            "irritation_streak": (
                rel.irritation_streak or 0.0
            ) if rel.last_message_at and (
                datetime.utcnow() - rel.last_message_at
            ).total_seconds() <= settings.irritation_window_seconds * 2 else 0.0,
            "relationship_type": rel.relationship_type or "friend",
            "is_canonical": bool(rel.is_canonical),
        }