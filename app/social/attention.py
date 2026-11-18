"""Dedicated attention/relevance mechanism.

Every incoming event receives an approximate attention score. Attention is
conceptual and deliberately independent of the participation score: it
measures *how much this event pulls Airi's interest*, while participation
measures *whether she acts on it*. High attention can:
  * pull her from AWAY into OBSERVING
  * unlock autonomous participation even without a mention
  * raise the chance she actually replies
Low attention lets her quietly ignore.

    attention =
        topic_interest
      + person_relevance
      + novelty
      + humor
      + emotional_salience
      + conflict
      + directness
      + relationship_relevance
      - repetition
      - boredom
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.analysis import MessageAnalysis, MessageTarget
from app.config import settings


@dataclass
class AttentionFactors:
    topic_interest: float = 0.0
    person_relevance: float = 0.0
    novelty: float = 0.0
    humor: float = 0.0
    emotional_salience: float = 0.0
    conflict: float = 0.0
    directness: float = 0.0
    relationship_relevance: float = 0.0
    repetition: float = 0.0
    boredom: float = 0.0
    attention_score: float = 0.0


# Mirrors Airi's personality: high for her primary interests, low for the
# "boring science" and music she irrationally dislikes.
BASE_TOPIC_INTEREST = {
    "minecraft": 0.9,
    "valorant": 0.9,
    "game": 0.7,
    "gaming": 0.7,
    "meme": 0.8,
    "brainrot": 0.8,
    "science": 0.15,
    "physics": 0.1,
    "math": 0.2,
    "music": 0.1,
    "song": 0.1,
}


class AttentionScorer:
    def score(
        self,
        analysis: MessageAnalysis,
        target: MessageTarget,
        topic_interest: Dict[str, float],
        relationship: Optional[Dict[str, float]],
        emotional_state: Optional[Dict[str, float]],
        current_topic: Optional[str] = None,
        obsessions: Optional[Dict[str, float]] = None,
    ) -> AttentionFactors:
        obsessions = obsessions or {}

        topic_interest_score = self._topic_interest(analysis, topic_interest, obsessions)

        person_relevance = self._person_relevance(target, relationship)

        # Canonical family (father): a real baseline attention advantage — not
        # an override. He can still be ignored over a boring topic, but he
        # starts meaningfully higher than strangers.
        is_anchor = bool(relationship and relationship.get("relationship_type") == "anchor")
        if is_anchor:
            person_relevance = max(person_relevance, 0.7)

        novelty = analysis.novelty

        humor = analysis.humor_likelihood

        emotional_salience = analysis.emotional_salience

        conflict = 0.0
        t = analysis.raw_text.lower()
        for marker in ("no,", "nah", "wrong", "false", "actually", "cap", "غلط", "نه", "اشتباه", "nuh uh", "bad take"):
            if marker in t:
                conflict = 0.6
                break

        directness = {
            MessageTarget.DIRECTLY_TO_AIRI: 1.0,
            MessageTarget.REPLY_TO_ANOTHER_USER: 0.3,
            MessageTarget.TO_ANOTHER_USER: 0.3,
            MessageTarget.GENERAL_GROUP_MESSAGE: 0.15,
            MessageTarget.UNCLEAR: 0.1,
        }.get(target, 0.1)

        rel = relationship or {}
        relationship_relevance = (
            rel.get("affection", 0.0) * 0.5
            + rel.get("familiarity", 0.0) * 0.3
            + max(rel.get("interest", 0.0), 0.0) * 0.2
        )
        relationship_relevance = max(0.0, min(1.0, relationship_relevance))

        # Repetition: novel messages get a small bump; heavily repeated chatter
        # loses a little. Handled primarily by the novelty field.
        repetition = max(0.0, 1.0 - novelty)

        boredom = float((emotional_state or {}).get("boredom", 0.2))

        # Questions pull humans in — someone asking something into the group
        # is a natural engagement hook, mention or not.
        question_pull = 0.15 if analysis.is_question else 0.0

        # Topic drift guard: if this message matches Airi's learned interests but
        # the conversation is on a different topic, attention still rises, but the
        # social decision layer will weigh "is it worth derailing for".
        attention = (
            topic_interest_score * 1.0
            + person_relevance * 0.6
            + novelty * 0.7
            + humor * 0.6
            + emotional_salience * 0.5
            + conflict * 0.5
            + directness * 0.8
            + relationship_relevance * 0.5
            + question_pull
            - repetition * 0.2
            - boredom * 0.6
        )

        return AttentionFactors(
            topic_interest=topic_interest_score,
            person_relevance=person_relevance,
            novelty=novelty,
            humor=humor,
            emotional_salience=emotional_salience,
            conflict=conflict,
            directness=directness,
            relationship_relevance=relationship_relevance,
            repetition=repetition,
            boredom=boredom,
            # Normalized to 0..1: the raw weighted sum can reach ~5, so scale
            # it down. All downstream thresholds (0.28 ignore / 0.55
            # participate / 0.60 surge) are defined on a 0..1 scale.
            attention_score=max(0.0, min(1.0, attention / 2.5)),
        )

    def _topic_interest(
        self,
        analysis: MessageAnalysis,
        topic_interest: Dict[str, float],
        obsessions: Dict[str, float],
    ) -> float:
        """Combine learned topic-interest values, obsessions, and keyword seeds."""
        text = analysis.raw_text.lower()
        keywords = [k.lower() for k in analysis.keywords]

        score = 0.0
        matched = False

        for kw in keywords:
            if kw in BASE_TOPIC_INTEREST:
                score = max(score, BASE_TOPIC_INTEREST[kw])
                matched = True

        for topic, interest in topic_interest.items():
            tl = topic.lower()
            if tl and (tl in text or any(tl in k or k in tl for k in keywords)):
                score = max(score, interest)
                matched = True

        for topic, engagement in obsessions.items():
            tl = topic.lower()
            if tl and (tl in text or any(tl in k or k in tl for k in keywords)):
                score = max(score, min(1.0, 0.5 + engagement * 0.05))
                matched = True

        if not matched:
            score = 0.2  # unknown topics: mild, not zero
        return max(0.0, min(1.0, score))

    def _person_relevance(self, target: MessageTarget, relationship: Optional[Dict[str, float]]) -> float:
        if relationship is None:
            return 0.1
        affection = relationship.get("affection", 0.0)
        if affection > 0.5:
            return 0.7
        if affection > 0.2:
            return 0.4
        return 0.2


attention_scorer = AttentionScorer()