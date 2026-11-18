from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, ForeignKey, Enum, JSON, Index, Boolean
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
import uuid

Base = declarative_base()


class MemoryType(PyEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    RELATIONSHIP = "relationship"
    PREFERENCE = "preference"
    CONVERSATION_SUMMARY = "conversation_summary"
    BEHAVIORAL_PATTERN = "behavioral_pattern"


class ConversationState(PyEnum):
    AWAY = "away"
    OBSERVING = "observing"
    INTERESTED = "interested"
    PARTICIPATING = "participating"
    WINDING_DOWN = "winding_down"


class ActionType(PyEnum):
    REPLY = "reply"
    IGNORE = "ignore"
    REACT = "react"
    INITIATE = "initiate"
    LEAVE = "leave"
    WAIT = "wait"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(Integer, unique=True, nullable=False, index=True)
    username = Column(String(255), nullable=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    memories = relationship("Memory", back_populates="user")
    relationships = relationship("Relationship", back_populates="user", cascade="all, delete-orphan")
    conversation_participants = relationship("ConversationParticipant", back_populates="user")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_chat_id = Column(Integer, unique=True, nullable=False, index=True)
    chat_type = Column(String(50), nullable=False)  # private, group, supergroup
    title = Column(String(500), nullable=True)
    last_activity_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    participants = relationship("ConversationParticipant", back_populates="conversation", cascade="all, delete-orphan")
    state = relationship("ConversationStateModel", back_populates="conversation", uselist=False, cascade="all, delete-orphan")
    summaries = relationship("ConversationSummary", back_populates="conversation", cascade="all, delete-orphan")


class ConversationParticipant(Base):
    __tablename__ = "conversation_participants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    joined_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    message_count = Column(Integer, default=0)

    conversation = relationship("Conversation", back_populates="participants")
    user = relationship("User", back_populates="conversation_participants")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    telegram_message_id = Column(Integer, nullable=False)
    text = Column(Text, nullable=True)
    reply_to_message_id = Column(Integer, nullable=True)
    # Canonical identity of the user being replied to (internal user id).
    # Reply chains must never lose who the original sender was.
    reply_to_user_id = Column(Integer, nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    is_from_bot = Column(Boolean, default=False)

    conversation = relationship("Conversation", back_populates="messages")
    user = relationship("User")


class Memory(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True, index=True)
    memory_type = Column(Enum(MemoryType), nullable=False, index=True)
    content = Column(Text, nullable=False)
    people_involved = Column(SQLiteJSON, default=list)  # list of user telegram_ids
    topics = Column(SQLiteJSON, default=list)  # list of topic strings
    importance = Column(Float, default=0.5, index=True)
    confidence = Column(Float, default=1.0)
    source = Column(String(100), nullable=False)  # "extraction", "observation", "correction", "summary"
    embedding = Column(SQLiteJSON, nullable=True)  # stored as list of floats
    last_retrieved = Column(DateTime, nullable=True)
    retrieval_count = Column(Integer, default=0)
    reinforcement = Column(Integer, default=0)  # times this fact was reconfirmed
    superseded_by = Column(Integer, ForeignKey("memories.id"), nullable=True)
    valid_from = Column(DateTime, nullable=True)  # for temporal memories
    valid_until = Column(DateTime, nullable=True)  # for temporal memories
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="memories")
    conversation = relationship("Conversation")
    superseding_memory = relationship("Memory", remote_side=[id], backref="superseded_memories")


class Relationship(Base):
    __tablename__ = "relationships"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    affection = Column(Float, default=0.0)  # -1 to 1
    trust = Column(Float, default=0.0)  # -1 to 1
    respect = Column(Float, default=0.0)  # -1 to 1
    annoyance = Column(Float, default=0.0)  # 0 to 1
    familiarity = Column(Float, default=0.0)  # 0 to 1
    interest = Column(Float, default=0.0)  # -1 to 1
    comfort = Column(Float, default=0.0)  # 0 to 1 (added for Airi)
    jealousy = Column(Float, default=0.0)  # 0 to 1 (added for Airi)
    # Canonical, config-driven social role (e.g. "anchor"), set once and never
    # overwritten by learned interaction data. is_canonical protects it from
    # decay and from contradictory learned updates.
    relationship_type = Column(String(30), default="friend", nullable=True)
    is_canonical = Column(Boolean, default=False)
    last_interaction = Column(DateTime, nullable=True)
    interaction_count = Column(Integer, default=0)
    # Short-term ping fatigue: how relentlessly this person has been messaging
    # her RECENTLY. Humans get briefly tired of one person spamming them —
    # separate from the slow, permanent `annoyance`. Decays on its own via
    # time-gap resets (see RelationshipManager.register_message_pulse).
    irritation_streak = Column(Float, default=0.0)
    last_message_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="relationships")


class ConversationStateModel(Base):
    __tablename__ = "conversation_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, unique=True, index=True)
    state = Column(Enum(ConversationState), default=ConversationState.AWAY, index=True)
    current_topic = Column(String(500), nullable=True)
    # A candidate new topic that has been seen ONCE but not yet confirmed.
    # A second consecutive sighting promotes it to current_topic; this stops
    # a single off-topic message from instantly derailing the conversation.
    pending_topic = Column(String(500), nullable=True)
    subtopic = Column(String(500), nullable=True)
    recent_topics = Column(SQLiteJSON, default=list)  # list of {topic, subtopic, at, transition_reason}
    recent_participants = Column(SQLiteJSON, default=list)  # list of user telegram_ids
    conversation_age_seconds = Column(Float, default=0.0)
    activity_level = Column(Float, default=0.0)
    is_participating = Column(Boolean, default=False)
    is_observing = Column(Boolean, default=False)
    winding_down_score = Column(Float, default=0.0)
    last_bot_message_at = Column(DateTime, nullable=True)
    last_bot_action = Column(Enum(ActionType), nullable=True)
    messages_since_bot = Column(Integer, default=0)
    initiations_without_engagement = Column(Integer, default=0)
    last_initiation_at = Column(DateTime, nullable=True)
    # Verbosity streak: which "typing mode" she's currently in for this
    # conversation (talkative / normal / low_effort) and when it expires.
    # A human stays in a mode for a stretch — several messages — before
    # drifting; independent per-message sampling reads as a slot machine.
    typing_register = Column(String(20), default="normal")
    typing_mode_until = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="state")


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    summary_text = Column(Text, nullable=False)
    topics = Column(SQLiteJSON, default=list)
    participants = Column(SQLiteJSON, default=list)
    message_count = Column(Integer, default=0)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    importance = Column(Float, default=0.5)
    embedding = Column(SQLiteJSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="summaries")


class EmotionalState(Base):
    __tablename__ = "emotional_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mood = Column(Float, default=0.0)  # -1 to 1 (negative to positive)
    energy = Column(Float, default=0.5)  # 0 to 1
    curiosity = Column(Float, default=0.5)  # 0 to 1
    amusement = Column(Float, default=0.0)  # 0 to 1
    annoyance = Column(Float, default=0.0)  # 0 to 1
    interest = Column(Float, default=0.5)  # 0 to 1
    social_engagement = Column(Float, default=0.5)  # 0 to 1
    confidence = Column(Float, default=0.5)  # 0 to 1 (added for Airi)
    embarrassment = Column(Float, default=0.0)  # 0 to 1 (added for Airi)
    competitiveness = Column(Float, default=0.5)  # 0 to 1 (added for Airi)
    boredom = Column(Float, default=0.2)  # 0 to 1 (added: stateful boredom)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class LanguageStats(Base):
    """Per-conversation output-language accounting.

    Tracks how overwhelmingly English Airi's output is, and how often she
    deliberately chose Persian. Used for observability and to keep the
    'English is the overwhelming default' invariant measurable.
    """
    __tablename__ = "language_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, unique=True, index=True)
    english_responses = Column(Integer, default=0)
    persian_responses = Column(Integer, default=0)
    persian_snippets = Column(Integer, default=0)  # short Persian phrases inside English responses
    last_language = Column(String(20), default="english")
    last_persian_reason = Column(String(500), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    conversation = relationship("Conversation")


class SelfModel(Base):
    """Airi's self-development data: obsessions, recurring jokes, learned tendencies.

    Singleton table. Updated by the learning loop from real interactions.
    Not hardcoded personality — this is what Airi has *developed* over time.
    """
    __tablename__ = "self_model"

    id = Column(Integer, primary_key=True, autoincrement=True)
    obsessions = Column(SQLiteJSON, default=dict)  # {topic: engagement_score}
    recurring_jokes = Column(SQLiteJSON, default=list)  # [{phrase, language, origin, times_used}]
    learned_tendencies = Column(SQLiteJSON, default=list)  # strings, e.g. "argues about valorant tier lists"
    topic_engagement = Column(SQLiteJSON, default=dict)  # {topic: rolling engagement score}
    # Established opinions: {topic: stance}. Keeps her takes consistent
    # across days — flipping opinions for no reason breaks the illusion of
    # a continuous person.
    stances = Column(SQLiteJSON, default=dict)
    # Weekly self-style audit output: {alarm_text: {count, pct, checked_utc}}.
    # Drift she should correct about her OWN voice (overused openers, question
    # rate, length creep). Regenerated each audit run — never accumulates stale.
    style_alarms = Column(SQLiteJSON, default=dict)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WorkingMemoryEntry(Base):
    """Short-term working memory: the current social situation.

    Compact, auto-expiring representation of what is happening *right now* in
    a conversation — recent messages, topic/subtopic, participants, events,
    and Airi's own recent actions. Not long-term memory.
    """
    __tablename__ = "working_memory_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    entry_type = Column(String(50), nullable=False, index=True)  # message, topic_transition, event, action, join, leave
    payload = Column(SQLiteJSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, nullable=True, index=True)

    conversation = relationship("Conversation")


class ScheduledAction(Base):
    """A lightweight delayed action for future Airi behavior.

    Represented as: action_type, execute_at, conversation, reason, state.
    Examples: delayed_reaction, return_to_conversation, follow_up, delayed_reply.
    Waiting never blocks the Telegram event loop — the background loop
    processes due actions.
    """
    __tablename__ = "scheduled_actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True, index=True)
    action_type = Column(String(50), nullable=False, index=True)  # react, return, follow_up, reply
    execute_at = Column(DateTime, nullable=False, index=True)
    payload = Column(SQLiteJSON, default=dict)  # e.g. {target_message_id, emoji, text, user_id}
    state = Column(String(50), default="pending")  # pending, executed, cancelled
    reason = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)

    conversation = relationship("Conversation")


class TopicInterest(Base):
    """Airi's evolving per-topic interest (0..1).

    Starts from personality-derived seeds (already expressed in the scorer)
    and evolves *gradually* from repeated experience: engaging experiences
    nudge interest up, boring/negative experiences nudge it down.
    """
    __tablename__ = "topic_interests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic = Column(String(300), nullable=False, unique=True, index=True)
    interest = Column(Float, default=0.5)  # 0..1
    exposure_count = Column(Integer, default=0)
    last_seen_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SocialGraphEdge(Base):
    """Lightweight social graph: observed relationships between ANY two users.

    Covers Airi↔other and other↔other observations. closeness/bond describes
    how close the two people appear to be; polarity is warmth/antagonism.
    """
    __tablename__ = "social_graph_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_a_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    user_b_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    closeness = Column(Float, default=0.0)  # 0..1 how close they appear
    polarity = Column(Float, default=0.0)   # -1..1 warmth vs antagonism
    interaction_count = Column(Integer, default=0)
    last_seen_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RecentPhrase(Base):
    """Anti-repetition tracker: recent phrases/jokes/topics Airi has used.

    Used to (a) avoid repeating herself and (b) let the LLM know what she
    recently said. Expired rows are pruned by background cleanup. target_user_id
    and topic allow repetition to be judged contextually (the same phrase aimed
    at a different person / in a different context is not spam).
    """
    __tablename__ = "recent_phrases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True, index=True)
    phrase_type = Column(String(50), default="phrase")  # phrase, joke, topic, response_pattern
    content = Column(Text, nullable=False)
    target_user_id = Column(Integer, nullable=True, index=True)
    topic = Column(String(300), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    conversation = relationship("Conversation")


class TrainingSample(Base):
    """Curated-log collection for a future personality fine-tune.

    Every reply she actually sends is persisted with a compact snapshot of
    the conversation that produced it (the same representation the LLM saw).
    This is the raw material for distillation-style iterative training: collect ->
    human-curate (rating column) -> distill into exemplars or fine-tune.
    Pure bookkeeping — written inline, never read by the agent itself.
    """
    __tablename__ = "training_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True, index=True)
    trigger = Column(String(20), nullable=False)  # reply | initiate
    # Compact context snapshot: [{sender, text}] in chronological order.
    context_snapshot = Column(SQLiteJSON, default=list)
    response_text = Column(Text, nullable=False)
    action = Column(String(20), nullable=True)
    language = Column(String(20), default="english")
    # Human curation: null = unreviewed; 1 = keep for training, 0 = discard.
    rating = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    conversation = relationship("Conversation")


class AppState(Base):
    """Tiny key/value store for runtime flags that must survive restarts.

    Kept separate from Settings because pydantic-settings is process-static —
    a runtime toggle needs a persistent, mutable home. (Historically home of
    the retired evil-mode toggle; the stale row in live DBs is inert.)
    """
    __tablename__ = "app_state"

    key = Column(String(64), primary_key=True)
    value = Column(String(255), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Indexes for common queries
Index("ix_memories_user_type", Memory.user_id, Memory.memory_type)
Index("ix_memories_conversation_type", Memory.conversation_id, Memory.memory_type)
Index("ix_memories_importance_created", Memory.importance.desc(), Memory.created_at.desc())
Index("ix_messages_conversation_timestamp", Message.conversation_id, Message.timestamp.desc())
Index("ix_conversation_state_state", ConversationStateModel.state)
Index("ix_working_memory_conv_expires", WorkingMemoryEntry.conversation_id, WorkingMemoryEntry.expires_at)
Index("ix_scheduled_actions_execute", ScheduledAction.execute_at, ScheduledAction.state)
Index("ix_social_graph_pair", SocialGraphEdge.user_a_id, SocialGraphEdge.user_b_id)
Index("ix_recent_phrases_conv_created", RecentPhrase.conversation_id, RecentPhrase.created_at)
Index("ix_training_samples_rating_created", TrainingSample.rating, TrainingSample.created_at)