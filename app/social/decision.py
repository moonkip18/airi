from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import random
import structlog

from pydantic import BaseModel, Field
from app.config import settings
from app.database import ActionType

logger = structlog.get_logger()


# ============================================================================
# ACTIONS
# ============================================================================

class SocialAction(str, Enum):
    REPLY = "reply"
    IGNORE = "ignore"
    REACT = "react"
    INITIATE = "initiate"
    LEAVE = "leave"
    WAIT = "wait"


class SocialDecisionMixin:
    """Shared helpers for social decision objects."""


class SocialDecision(BaseModel):
    action: SocialAction
    target_user: Optional[int] = None
    reasoning_summary: str
    current_topic: str
    confidence: float = Field(ge=0.0, le=1.0)
    response: Optional[str] = None
    # Optional follow-up she fires off ~30s later if nobody reacted to her
    # reply ("wait", "also", one extra thought). Usually empty — the send
    # side only delivers it when the chat stayed quiet, so it's a real
    # double-text, never an interruption.
    double_text: str = ""
    continue_participating: bool = True
    memory_ids_used: List[str] = []
    response_delay: float = 0.0
    language: str = "english"
    persian_snippet: bool = False
    reaction_emoji: Optional[str] = None
    wait_until: Optional[str] = None


# ============================================================================
# MESSAGE MARKERS
# ============================================================================

DISAGREEMENT_MARKERS = [
    "no,",
    "nah",
    "wrong",
    "not true",
    "false",
    "actually",
    "dumb",
    "stupid",
    "bad take",
    "cap",
    "lying",
    "nuh uh",
    "shut up",
    "cope",
    "ratio",
    "غلط",
    "نه",
    "اشتباه",
    "بیشعور",
    "خفه",
    "بی‌خود",
]

QUESTION_MARKERS = ["?", "؟"]


# ============================================================================
# FACTOR GROUPS
# ============================================================================

@dataclass
class AttentionFactors:
    """
    Things that make Airi mentally notice/care about a message.

    These describe attraction toward the message itself.
    They should NOT contain inhibition factors such as boredom or saturation.
    """

    topic_relevance: float = 0.0
    personal_interest: float = 0.0
    relationship_relevance: float = 0.0
    conversation_energy: float = 0.0
    curiosity: float = 0.0
    emotional_salience: float = 0.0
    novelty: float = 0.0
    humor: float = 0.0
    directness: float = 0.0


@dataclass
class SocialSignals:
    """
    Social cues that can create pressure to act.
    """

    direct_mention: float = 0.0
    is_disagreement: bool = False
    competitiveness: float = 0.5
    jealousy: float = 0.0
    mentions_interest: bool = False
    target_is_important: float = 0.0


@dataclass
class SocialConstraints:
    """
    Things that suppress participation.
    """

    recent_activity: float = 0.0
    conversation_saturation: float = 0.0
    bot_reply_burst: float = 0.0
    boredom: float = 0.2
    annoyance: float = 0.0
    embarrassment: float = 0.0
    low_energy: float = 0.0


@dataclass
class ParticipationFactors:
    """
    Backwards-compatible factor object.

    Existing Agent code expects this class, so the original fields remain.
    Internally the engine now treats them as separate signal groups.
    """

    direct_mention: float = 0.0
    topic_relevance: float = 0.0
    personal_interest: float = 0.0
    relationship_relevance: float = 0.0
    conversation_energy: float = 0.0
    curiosity: float = 0.0
    current_mood: float = 0.0
    recent_activity: float = 0.0
    conversation_saturation: float = 0.0
    is_autonomous: bool = False

    energy_state: float = 0.5
    annoyance_state: float = 0.0
    competitiveness: float = 0.5
    embarrassment: float = 0.0
    jealousy: float = 0.0
    boredom: float = 0.2
    attention: float = 0.0
    bot_reply_burst: float = 0.0
    is_disagreement: bool = False
    mentions_interest: bool = False

    # Extra local signals.
    emotional_salience: float = 0.0
    novelty: float = 0.0
    humor: float = 0.0
    directness: float = 0.0


@dataclass
class ActionScores:
    """
    Competition between possible social actions.

    The important architectural change is that Airi no longer asks only:

        "Should I participate?"

    She asks:

        "What action has the strongest social pressure right now?"
    """

    reply: float = 0.0
    react: float = 0.0
    wait: float = 0.0
    ignore: float = 0.0
    leave: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "reply": self.reply,
            "react": self.react,
            "wait": self.wait,
            "ignore": self.ignore,
            "leave": self.leave,
        }


# ============================================================================
# SOCIAL DECISION ENGINE
# ============================================================================

class SocialDecisionEngine:

    # Only emojis from Telegram's allowed reaction set — anything else gets
    # rejected by the API (😂/💀/😮 look natural but are NOT allowed).
    REACTION_EMOJIS_FUNNY = ["🤣", "😭", "🤡", "🤪"]
    REACTION_EMOJIS_WOW = ["👀", "😱", "🤯", "🤩"]
    REACTION_EMOJIS_LOVE = ["❤", "😍", "🥰", "🍓", "💘", "💯", "🔥"]
    REACTION_EMOJIS_NEUTRAL = ["👍", "👌"]

    VALID_REACTIONS = set(
        REACTION_EMOJIS_FUNNY
        + REACTION_EMOJIS_WOW
        + REACTION_EMOJIS_LOVE
        + REACTION_EMOJIS_NEUTRAL
    )

    def __init__(self):
        self.message_timestamps: List[float] = []
        # Reaction cooldown (wall-clock): reactions are a quiet gesture, and
        # stacking several within a couple of minutes reads as a bot mashing
        # the emoji button. One reaction per conversation per ~90s.
        self._last_reaction_at: Dict[int, float] = {}

    def reaction_on_cooldown(self, conversation_id: int, seconds: float = 90.0) -> bool:
        import time as _time
        last = self._last_reaction_at.get(conversation_id)
        return last is not None and (_time.time() - last) < seconds

    def record_reaction(self, conversation_id: int):
        import time as _time
        self._last_reaction_at[conversation_id] = _time.time()

    # ----------------------------------------------------------------------
    # NORMALIZATION
    # ----------------------------------------------------------------------

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _safe(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # ----------------------------------------------------------------------
    # ATTENTION
    # ----------------------------------------------------------------------

    def calculate_attention(self, factors: ParticipationFactors) -> float:
        """
        Calculate how strongly the message captures Airi's attention.

        This is deliberately deterministic.

        Randomness does NOT belong here because attention is an internal
        measurable state. Randomness is applied later when competing actions
        are close.
        """

        # Core semantic attention.
        semantic = (
            factors.topic_relevance * 0.22 +
            factors.personal_interest * 0.20 +
            factors.relationship_relevance * 0.13 +
            factors.conversation_energy * 0.05 +
            factors.curiosity * 0.08 +
            factors.current_mood * 0.03 +
            factors.emotional_salience * 0.08 +
            factors.novelty * 0.08 +
            factors.humor * 0.08 +
            factors.directness * 0.05 +
            # Being addressed directly is itself attention-grabbing.
            factors.direct_mention * 0.30
        )

        # Super-linear response to highly engaging content.
        semantic = self._clamp(semantic)

        # Preserve the externally calculated attention if another subsystem
        # has already supplied one. This makes the engine backwards compatible
        # with the attention scorer.
        supplied_attention = self._clamp(factors.attention)

        if supplied_attention > 0.0:
            # The externally calculated attention dominates; local semantic
            # information refines it. (A heavy local blend was dragging every
            # non-mention score into the floor.)
            combined = (
                supplied_attention * 0.80 +
                semantic * 0.20
            )
        else:
            combined = semantic

        return self._clamp(combined)

    # ----------------------------------------------------------------------
    # SOCIAL PRESSURE
    # ----------------------------------------------------------------------

    def calculate_social_pressure(
        self,
        factors: ParticipationFactors,
    ) -> float:
        """
        How strongly social circumstances push Airi toward action.
        """

        pressure = 0.0

        # Direct address is important, but no longer an automatic reply.
        pressure += factors.direct_mention * 0.28

        # Important interests create additional pull.
        if factors.mentions_interest:
            pressure += 0.08

        # Mild jealousy can make attention toward a person more salient.
        pressure += factors.jealousy * 0.06

        # Disagreement activates competitiveness.
        if factors.is_disagreement:
            pressure += factors.competitiveness * 0.09

        return self._clamp(pressure)

    # ----------------------------------------------------------------------
    # INHIBITION
    # ----------------------------------------------------------------------

    def calculate_inhibition(
        self,
        factors: ParticipationFactors,
    ) -> float:
        """
        Calculate forces telling Airi NOT to participate.
        """

        inhibition = 0.0

        # Recent activity means she has already been socially active.
        inhibition += factors.recent_activity * 0.12

        # Conversation saturation.
        inhibition += factors.conversation_saturation * 0.15

        # Strong burst detection.
        inhibition += factors.bot_reply_burst * 0.42

        # Boredom is a real suppression signal.
        inhibition += factors.boredom * 0.16

        # Annoyance suppresses participation.
        inhibition += factors.annoyance_state * 0.10

        # Embarrassment suppresses action.
        inhibition += factors.embarrassment * 0.07

        # Low energy.
        low_energy = 1.0 - self._clamp(factors.energy_state)
        inhibition += low_energy * 0.08

        return self._clamp(inhibition)

    # ----------------------------------------------------------------------
    # PARTICIPATION SCORE
    # ----------------------------------------------------------------------

    def calculate_participation_score(
        self,
        factors: ParticipationFactors,
    ) -> float:
        """
        Backwards-compatible scalar participation score.

        This is now derived from three conceptual layers:

            attention
            + social pressure
            - inhibition

        rather than one giant undifferentiated weighted equation.
        """

        attention = self.calculate_attention(factors)
        social_pressure = self.calculate_social_pressure(factors)
        inhibition = self.calculate_inhibition(factors)

        # Attention is mildly nonlinear — enough to separate "mildly noticed"
        # from "genuinely engaged" without crushing mid-range attention into
        # the floor (the old 1.35 exponent made 0.9 attention score 0.55).
        attention_pull = attention ** 1.20

        score = (
            attention_pull * 0.65 +
            social_pressure * 0.40 -
            inhibition * 0.35
        )

        # NOTE: no autonomous penalty multiplier here. The old *0.40 factor
        # capped the theoretical maximum below the autonomous threshold,
        # making uninvited participation mathematically impossible. The
        # "harder for autonomous" requirement is enforced at the gate in
        # Agent.autonomous_check instead.

        return self._clamp(score)

    # ----------------------------------------------------------------------
    # ACTION SCORING
    # ----------------------------------------------------------------------

    def score_actions(
        self,
        factors: ParticipationFactors,
    ) -> ActionScores:
        """
        Score every available social action.

        This is the main decision architecture.

        A message can be:
            interesting but reaction-worthy
            interesting but not reply-worthy
            socially important enough to reply
            mildly interesting enough to wait
            completely irrelevant
            boring enough to leave
        """

        attention = self.calculate_attention(factors)
        pressure = self.calculate_social_pressure(factors)
        inhibition = self.calculate_inhibition(factors)

        # ------------------------------------------------------------------
        # REPLY
        # ------------------------------------------------------------------

        reply = (
            attention * 0.58 +
            pressure * 0.42
        )

        # Direct questions should strongly favor replying.
        if factors.direct_mention:
            reply += 0.08

        # Questions are implicitly represented by emotional/semantic signals
        # in the message-analysis layer, but directness remains the strongest
        # local signal.

        # Strong inhibition makes a reply harder.
        reply -= inhibition * 0.65

        # ------------------------------------------------------------------
        # REACTION
        # ------------------------------------------------------------------

        reaction = (
            attention * 0.48 +
            factors.humor * 0.28 +
            factors.emotional_salience * 0.18 +
            pressure * 0.06
        )

        # Reactions are intentionally cheaper than replies.
        reaction -= inhibition * 0.30

        # ------------------------------------------------------------------
        # WAIT
        # ------------------------------------------------------------------

        # WAIT is strongest when she notices something but the environment
        # isn't good for immediate participation.
        moderate_attention = 1.0 - abs(attention - 0.55) / 0.55
        moderate_attention = self._clamp(moderate_attention)

        wait = (
            moderate_attention * 0.45 +
            attention * 0.25 +
            inhibition * 0.20
        )

        # If she's extremely interested, waiting becomes less attractive.
        if attention >= 0.80:
            wait -= 0.18

        # ------------------------------------------------------------------
        # IGNORE
        # ------------------------------------------------------------------

        ignore = (
            (1.0 - attention) * 0.58 +
            inhibition * 0.30 +
            factors.boredom * 0.12
        )

        # Strong direct address makes ignoring less likely.
        ignore -= factors.direct_mention * 0.22

        # ------------------------------------------------------------------
        # LEAVE
        # ------------------------------------------------------------------

        leave = (
            factors.boredom * 0.52 +
            factors.annoyance_state * 0.22 +
            factors.conversation_saturation * 0.16 +
            factors.bot_reply_burst * 0.10
        )

        # Leave should not happen merely because of one boring message.
        leave *= 0.85

        return ActionScores(
            reply=self._clamp(reply),
            react=self._clamp(reaction),
            wait=self._clamp(wait),
            ignore=self._clamp(ignore),
            leave=self._clamp(leave),
        )

    # ----------------------------------------------------------------------
    # ACTION SELECTION
    # ----------------------------------------------------------------------

    def _action_probability(
        self,
        score: float,
        threshold: float,
        randomness: float = 0.15,
    ) -> float:
        """
        Convert score proximity to threshold into probability.

        Deterministic far above/below threshold.
        Stochastic near the boundary.
        """

        if score >= threshold + randomness:
            return 1.0

        if score <= threshold - randomness:
            return 0.0

        return self._clamp(
            0.5 + (score - threshold) / (2.0 * randomness)
        )

    def choose_action(
        self,
        factors: ParticipationFactors,
        *,
        is_direct_mention: Optional[bool] = None,
        is_autonomous: Optional[bool] = None,
        rng: Optional[random.Random] = None,
    ) -> SocialAction:
        """
        Select an action from the action competition.

        The result is deterministic for clearly dominant actions and mildly
        stochastic near boundaries.
        """

        rng = rng or random

        if is_direct_mention is None:
            is_direct_mention = factors.direct_mention >= 0.5

        if is_autonomous is None:
            is_autonomous = factors.is_autonomous

        scores = self.score_actions(factors)

        # --------------------------------------------------------------
        # HARD SAFETY / SOCIAL CAPS
        # --------------------------------------------------------------

        if factors.bot_reply_burst >= 0.85:
            scores.reply *= 0.35
            scores.react *= 0.65

        if factors.recent_activity >= 0.85:
            scores.reply *= 0.55

        if factors.conversation_saturation >= 0.90:
            scores.reply *= 0.45

        # Autonomous participation cannot simply decide to reply because
        # the local action scorer liked the message.
        if is_autonomous:
            scores.reply *= 0.80
            scores.react *= 0.65

        # --------------------------------------------------------------
        # LEAVE
        # --------------------------------------------------------------

        if (
            factors.boredom >= settings.boredom_leave_threshold
            and scores.leave >= 0.52
            and not is_direct_mention
        ):
            return SocialAction.LEAVE

        # --------------------------------------------------------------
        # DIRECT MENTION
        # --------------------------------------------------------------

        if is_direct_mention:
            # A direct mention strongly favors action, but does not force
            # a reply when the message is trivial.
            if scores.reply >= 0.40:
                return SocialAction.REPLY

            if scores.react >= 0.40:
                return SocialAction.REACT

            return SocialAction.WAIT

        # --------------------------------------------------------------
        # AUTONOMOUS
        # --------------------------------------------------------------

        if is_autonomous:
            participation = self.calculate_participation_score(factors)

            if participation < settings.autonomous_participation_threshold:
                return SocialAction.IGNORE

            if scores.reply >= 0.55:
                return SocialAction.INITIATE

            if scores.react >= 0.45:
                return SocialAction.REACT

            return SocialAction.WAIT

        # --------------------------------------------------------------
        # NORMAL ACTION COMPETITION
        # --------------------------------------------------------------

        values = scores.as_dict()

        # Don't let LEAVE win during ordinary low-boredom situations.
        if factors.boredom < settings.boredom_leave_threshold:
            values["leave"] *= 0.25

        ranked = sorted(
            values.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        best_action, best_score = ranked[0]
        second_action, second_score = ranked[1]

        # Very clear winner.
        if best_score - second_score >= 0.16:
            return SocialAction(best_action)

        # Close competition: introduce mild stochasticity.
        candidates = [
            (name, score)
            for name, score in ranked
            if score >= best_score - 0.10
        ]

        total = sum(max(score, 0.01) for _, score in candidates)

        roll = rng.random() * total

        cumulative = 0.0
        for name, score in candidates:
            cumulative += max(score, 0.01)
            if roll <= cumulative:
                return SocialAction(name)

        return SocialAction(best_action)

    # ----------------------------------------------------------------------
    # LEGACY PARTICIPATION GATE
    # ----------------------------------------------------------------------

    def should_participate(
        self,
        score: float,
        is_direct_mention: bool,
        is_autonomous: bool,
        attention: float = 0.0,
        boredom: float = 0.2,
        recent_activity: float = 0.0,
        conversation_saturation: float = 0.0,
        bot_reply_burst: float = 0.0,
    ) -> bool:
        """
        Backwards-compatible gate for Agent.

        This remains available because your current Agent uses it.

        Internally it now behaves like a conservative participation gate
        rather than being the entire decision architecture.
        """

        # Hard burst cap.
        if bot_reply_burst >= 0.90:
            return False

        # Autonomous uses its own threshold.
        if is_autonomous:
            return score >= settings.autonomous_participation_threshold

        # Direct mentions are easier to answer, but not automatic.
        if is_direct_mention:
            if attention >= 0.55:
                return True
            return score >= 0.20

        # --- Suppression guards: a bored, saturated, or over-active Airi
        # stays quiet no matter how decent the raw score is. Only a direct
        # mention (handled above) cuts through these.
        if boredom >= settings.boredom_threshold:
            return False
        if conversation_saturation >= 0.70:
            return False
        if recent_activity >= 0.70:
            return False

        # Clearly strong involvement participates deterministically.
        if score >= 0.40:
            return True

        # Below that, attention decides stochastically: the more caught she
        # is, the likelier she chimes in. Boredom dampens.
        if attention >= 0.38:
            surge_probability = (
                0.32 +
                0.45 * ((attention - 0.38) / 0.62)
            )

            surge_probability *= max(0.0, 1.0 - boredom)

            return random.random() < self._clamp(surge_probability)

        return False

    # ----------------------------------------------------------------------
    # PASSIVE ACTION
    # ----------------------------------------------------------------------

    def decide_passive_action(
        self,
        score: float,
        attention: float,
        boredom: float,
        want_to_leave: bool = False,
    ) -> SocialAction:

        if want_to_leave:
            return SocialAction.LEAVE

        # Very low attention = genuine ignore.
        if attention < 0.28:
            return SocialAction.IGNORE

        # High boredom suppresses social engagement.
        if boredom >= settings.boredom_threshold:
            return SocialAction.IGNORE

        # Moderate attention without enough action pressure = WAIT.
        if attention >= 0.42 and score < 0.42:
            return SocialAction.WAIT

        return SocialAction.IGNORE

    # ----------------------------------------------------------------------
    # REACTIONS
    # ----------------------------------------------------------------------

    def decide_reaction(
        self,
        attention: float,
        humor: float = 0.0,
        emotional_salience: float = 0.0,
        boredom: float = 0.2,
        recent_activity: float = 0.0,
        conversation_saturation: float = 0.0,
        positive: Optional[bool] = None,
        is_question: bool = False,
        conversation_hot: bool = False,
        rng: Optional[random.Random] = None,
    ) -> Optional[str]:
        """
        Decide whether a message deserves a lightweight reaction.

        Reactions are the "nothing to add" gesture: for when the topic has
        basically run its course or the message was cute/funny but a reply
        would be forced. They are NOT a substitute for talking while the
        conversation is alive — that's what replies are for.

        In a HOT conversation a reaction is how she stays present without
        adding noise (or stealing the spotlight) when the reply gate didn't
        open — so it's allowed there, just at reduced probability.
        """

        rng = rng or random

        if boredom >= settings.boredom_threshold:
            return None

        if recent_activity >= 0.78:
            return None

        if conversation_saturation >= 0.82:
            return None

        # Questions deserve words. Reacting to a question reads as dismissive.
        if is_question:
            return None

        # High attention means she has something to say — reply territory.
        # Reactions live in the mild-interest band only.
        if not (0.30 <= attention < 0.80):
            return None

        if humor < 0.25 and emotional_salience < 0.30:
            return None

        strength = max(humor, emotional_salience)

        # Rare by design: a quiet acknowledgment, not a habit.
        probability = 0.15 + 0.30 * strength

        # Emotional messages can produce stronger reactions.
        if emotional_salience >= 0.65:
            probability += 0.06

        # In a live conversation a reaction is a background gesture —
        # keep it clearly secondary to actually answering.
        if conversation_hot:
            probability *= 0.6

        probability = self._clamp(probability)

        if rng.random() >= probability:
            return None

        # Pick the FAMILY by vibe: funny beats everything, sweet/positive
        # messages get hearts (not shock), strong emotion without positivity
        # reads as "whoa", the rest is a quiet thumbs-up tier.
        if humor >= emotional_salience:
            return rng.choice(self.REACTION_EMOJIS_FUNNY)

        if positive is True:
            return rng.choice(self.REACTION_EMOJIS_LOVE)

        if emotional_salience >= 0.60:
            return rng.choice(self.REACTION_EMOJIS_WOW)

        return rng.choice(self.REACTION_EMOJIS_NEUTRAL)

    # ----------------------------------------------------------------------
    # RESPONSE DELAY
    # ----------------------------------------------------------------------

    def calculate_response_delay(
        self,
        is_direct_mention: bool,
        is_autonomous: bool,
        message_complexity: float = 0.5,
        conversation_pace: float = 0.5,
        energy: float = 0.5,
    ) -> float:

        if is_direct_mention:
            base = settings.direct_mention_delay
        elif is_autonomous:
            base = settings.autonomous_response_delay
        else:
            base = settings.min_response_delay

        message_complexity = self._clamp(message_complexity)
        conversation_pace = self._clamp(conversation_pace)
        energy = self._clamp(energy)

        # state-driven responsiveness: snappy by default, slower only for
        # genuinely long/complex messages. The old factors (complexity*2,
        # pace*1.0, energy*1.5) routinely produced 4-7s for one-liner banter,
        # which reads as dead air in a text chat.
        complexity_factor = 1.0 + message_complexity * 1.2
        pace_factor = 1.0 + conversation_pace * 0.6

        # Lower energy = slower response.
        energy_factor = 1.0 + (1.0 - energy) * 0.8

        delay = (
            base *
            complexity_factor *
            pace_factor *
            energy_factor
        )

        # Timing randomness belongs here, not in the semantic score.
        delay += random.uniform(-0.3, 0.9)

        return max(
            settings.min_response_delay,
            min(settings.max_response_delay, delay),
        )

    # ----------------------------------------------------------------------
    # RATE LIMITING
    # ----------------------------------------------------------------------

    def record_message(self):
        import time

        now = time.time()

        self.message_timestamps.append(now)

        self.message_timestamps = [
            timestamp
            for timestamp in self.message_timestamps
            if now - timestamp < 60
        ]

    def get_recent_message_rate(self) -> float:
        import time

        now = time.time()

        self.message_timestamps = [
            timestamp
            for timestamp in self.message_timestamps
            if now - timestamp < 60
        ]

        return len(self.message_timestamps)

    def is_rate_limited(self) -> bool:
        return (
            self.get_recent_message_rate()
            >= settings.max_messages_per_minute
        )

    # ----------------------------------------------------------------------
    # DISAGREEMENT
    # ----------------------------------------------------------------------

    @staticmethod
    def looks_like_disagreement(message_text: str) -> bool:
        text = (message_text or "").lower()

        return any(
            marker in text
            for marker in DISAGREEMENT_MARKERS
        )

    # ----------------------------------------------------------------------
    # DEBUG / EXPLANATION
    # ----------------------------------------------------------------------

    def explain_decision(
        self,
        factors: ParticipationFactors,
    ) -> Dict[str, Any]:
        """
        Useful for debugging and tuning the bot.

        This lets you see WHY an action was attractive rather than only seeing
        the final decision.
        """

        attention = self.calculate_attention(factors)
        pressure = self.calculate_social_pressure(factors)
        inhibition = self.calculate_inhibition(factors)
        participation = self.calculate_participation_score(factors)
        actions = self.score_actions(factors)

        return {
            "attention": round(attention, 3),
            "social_pressure": round(pressure, 3),
            "inhibition": round(inhibition, 3),
            "participation": round(participation, 3),
            "actions": {
                key: round(value, 3)
                for key, value in actions.as_dict().items()
            },
            "signals": {
                "direct_mention": round(factors.direct_mention, 3),
                "topic_relevance": round(factors.topic_relevance, 3),
                "personal_interest": round(factors.personal_interest, 3),
                "relationship_relevance": round(
                    factors.relationship_relevance, 3
                ),
                "humor": round(factors.humor, 3),
                "emotional_salience": round(
                    factors.emotional_salience, 3
                ),
                "novelty": round(factors.novelty, 3),
                "jealousy": round(factors.jealousy, 3),
                "boredom": round(factors.boredom, 3),
                "annoyance": round(factors.annoyance_state, 3),
                "burst": round(factors.bot_reply_burst, 3),
            },
        }


# ============================================================================
# SOCIAL SCORER
# ============================================================================

class SocialScorer:
    """
    Local message scoring.

    NO LLM calls.

    The scorer gathers raw signals. The SocialDecisionEngine is responsible
    for converting them into attention, inhibition and action pressure.
    """

    INTEREST_KEYWORDS = {
        "minecraft",
        "valorant",
        "game",
        "gaming",
        "gamer",
        "rank",
        "radiant",
        "immortal",
        "skin",
        "bundle",
        "meme",
        "memes",
        "brainrot",
        "67",
        "six seven",
        "skibidi",
        "rizz",
        "gyatt",
        "fanum",
        "ohio",
        "sigma",
        "goofy",
        "ragebait",
        "troll",
        "shitpost",
        "clip",
        "clutch",
        "ace",
        "fps",
    }

    def __init__(self, session):
        self.session = session

    async def score_message(
        self,
        message_text: str,
        user_id: int,
        conversation_context,
        emotional_state: Dict[str, float],
        relationship: Dict[str, float],
        retrieved_memories: List[Any],
        is_direct_mention: bool,
        bot_username: str,
        obsessions: Optional[Dict[str, float]] = None,
        attention: float = 0.0,
        message_analysis: Any = None,
    ) -> ParticipationFactors:

        direct_mention = 1.0 if is_direct_mention else 0.0

        topic_relevance = self._score_topic_relevance_local(
            message_text,
            conversation_context,
        )

        personal_interest = self._score_personal_interest(
            message_text,
            retrieved_memories,
            obsessions or {},
        )

        relationship_relevance = (
            self._score_relationship_relevance(relationship)
        )

        conversation_energy = (
            conversation_context.state.activity_level
            if conversation_context.state
            else 0.5
        )

        curiosity = emotional_state.get(
            "curiosity",
            0.5,
        )

        current_mood = (
            emotional_state.get("mood", 0.0) + 1.0
        ) / 2.0

        # "Recent activity" measures how much SHE has been dominating the
        # recent window — not how busy the chat is. (Counting total chat
        # volume made the suppression guard permanently trigger in any
        # active group, so she'd only ever reply to direct mentions.)
        recent_window = list(
            getattr(conversation_context, "recent_messages", []) or []
        )[-10:]

        own_share = (
            sum(
                1 for m in recent_window
                if getattr(m, "is_from_bot", False)
            )
            / max(len(recent_window), 1)
        )

        recent_activity = min(1.0, own_share * 2.0)

        conversation_saturation = min(
            1.0,
            conversation_context.bot_message_count
            / settings.conversation_saturation_threshold,
        )

        energy_state = emotional_state.get(
            "energy",
            0.5,
        )

        annoyance_state = emotional_state.get(
            "annoyance",
            0.0,
        )

        competitiveness = emotional_state.get(
            "competitiveness",
            0.5,
        )

        embarrassment = emotional_state.get(
            "embarrassment",
            0.0,
        )

        jealousy = (
            relationship.get("jealousy", 0.0)
            if relationship
            else 0.0
        )

        # --------------------------------------------------------------
        # BOT BURST
        # --------------------------------------------------------------

        recent = list(
            getattr(
                conversation_context,
                "recent_messages",
                [],
            ) or []
        )[-6:]

        bot_in_window = sum(
            1
            for message in recent
            if getattr(message, "is_from_bot", False)
        )

        burst = (
            bot_in_window / 8.0
            if recent
            else 0.0
        )

        last_bot_at = getattr(
            conversation_context,
            "last_bot_message_at",
            None,
        )

        if last_bot_at is not None:
            from datetime import datetime

            since = (
                datetime.utcnow() - last_bot_at
            ).total_seconds()

            if since < 25:
                burst = max(burst, 0.5)

        # --------------------------------------------------------------
        # MESSAGE SIGNALS
        # --------------------------------------------------------------

        is_disagreement = (
            SocialDecisionEngine.looks_like_disagreement(
                message_text
            )
        )

        mentions_interest = self._mentions_interest(
            message_text,
            obsessions or {},
        )

        boredom = emotional_state.get(
            "boredom",
            0.2,
        )

        # These are deliberately exposed independently instead of being
        # hidden inside the participation equation. When the caller supplies
        # the MessageAnalysis object we read the REAL local signals — humor,
        # novelty and emotional charge feed directly into reaction/reply
        # scoring instead of being flattened to zero.
        emotional_salience = self._analysis_signal(
            message_analysis, "emotional_salience", default=0.0,
        )

        novelty = self._analysis_signal(
            message_analysis, "novelty", default=0.0,
        )

        humor = self._analysis_signal(
            message_analysis, "humor_likelihood", default=0.0,
        )

        directness = direct_mention

        return ParticipationFactors(
            direct_mention=direct_mention,
            topic_relevance=topic_relevance,
            personal_interest=personal_interest,
            relationship_relevance=relationship_relevance,
            conversation_energy=conversation_energy,
            curiosity=curiosity,
            current_mood=current_mood,
            recent_activity=recent_activity,
            conversation_saturation=conversation_saturation,
            energy_state=energy_state,
            annoyance_state=annoyance_state,
            competitiveness=competitiveness,
            embarrassment=embarrassment,
            jealousy=jealousy,
            boredom=boredom,
            attention=attention,
            bot_reply_burst=burst,
            is_disagreement=is_disagreement,
            mentions_interest=mentions_interest,
            emotional_salience=emotional_salience,
            novelty=novelty,
            humor=humor,
            directness=directness,
        )

    # ----------------------------------------------------------------------
    # LOCAL TOPIC RELEVANCE
    # ----------------------------------------------------------------------

    def _score_topic_relevance_local(
        self,
        message_text: str,
        conversation_context,
    ) -> float:

        state = (
            conversation_context.state
            if conversation_context
            else None
        )

        if not state or not state.current_topic:
            return 0.5

        from app.memory.embeddings import embedding_model

        query_vec = embedding_model.encode_single(
            message_text
        )

        topic_vec = embedding_model.encode_single(
            state.current_topic
        )

        similarity = embedding_model.similarity(
            query_vec,
            topic_vec,
        )

        return max(
            0.0,
            min(1.0, similarity),
        )

    # ----------------------------------------------------------------------
    # PERSONAL INTEREST
    # ----------------------------------------------------------------------

    def _score_personal_interest(
        self,
        message_text: str,
        retrieved_memories: List[Any],
        obsessions: Dict[str, float],
    ) -> float:

        text = (message_text or "").lower()

        score = 0.2

        hits = sum(
            1
            for keyword in self.INTEREST_KEYWORDS
            if keyword in text
        )

        score += min(
            0.5,
            hits * 0.2,
        )

        for topic, engagement in obsessions.items():

            topic_words = topic.lower().split()

            if any(
                word and word in text
                for word in topic_words
            ):
                score += min(
                    0.4,
                    engagement * 0.05,
                )

        if retrieved_memories:

            interest_signals = sum(
                memory.relevance_score
                for memory in retrieved_memories
                if memory.memory.memory_type.value
                in [
                    "preference",
                    "semantic",
                    "behavioral_pattern",
                ]
            )

            score += min(
                0.3,
                interest_signals * 0.15,
            )

        return min(
            1.0,
            score,
        )

    # ----------------------------------------------------------------------
    # INTEREST DETECTION
    # ----------------------------------------------------------------------

    def _mentions_interest(
        self,
        message_text: str,
        obsessions: Dict[str, float],
    ) -> bool:

        text = (message_text or "").lower()

        if any(
            keyword in text
            for keyword in self.INTEREST_KEYWORDS
        ):
            return True

        for topic in obsessions:

            topic_words = topic.lower().split()

            if any(
                word and word in text
                for word in topic_words
            ):
                return True

        return False

    # ----------------------------------------------------------------------
    # RELATIONSHIP
    # ----------------------------------------------------------------------

    def _score_relationship_relevance(
        self,
        relationship: Dict[str, float],
    ) -> float:

        if not relationship:
            return 0.3

        affection = relationship.get(
            "affection",
            0.0,
        )

        trust = relationship.get(
            "trust",
            0.0,
        )

        annoyance = relationship.get(
            "annoyance",
            0.0,
        )

        familiarity = relationship.get(
            "familiarity",
            0.0,
        )

        interest = relationship.get(
            "interest",
            0.0,
        )

        base = (
            affection +
            trust +
            familiarity +
            max(interest, 0.0)
        ) / 4.0

        return max(
            0.0,
            min(
                1.0,
                base - annoyance * 0.5,
            ),
        )

    # ----------------------------------------------------------------------
    # MESSAGE ANALYSIS BRIDGE
    # ----------------------------------------------------------------------

    @staticmethod
    def _analysis_signal(
        analysis: Any,
        field: str,
        default: float = 0.0,
    ) -> float:
        """Read a numeric signal off the MessageAnalysis object when the
        caller supplied one; otherwise fall back to the default."""
        if analysis is None:
            return default
        try:
            value = getattr(analysis, field, None)
            if value is None:
                return default
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default