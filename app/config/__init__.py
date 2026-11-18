from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Telegram
    telegram_bot_token: str = Field(
        default="",
        alias="TELEGRAM_BOT_TOKEN",
    )

    telegram_proxy: Optional[str] = Field(
        default=None,
        alias="TELEGRAM_PROXY",
    )

    # LLM gateway (any OpenAI-compatible endpoint: OpenAI, Groq, OpenRouter,
    # a local Ollama/LM Studio server, or a commercial gateway)
    llm_api_key: str = Field(
        default="",
        alias="LLM_API_KEY",
    )

    llm_base_url: str = Field(
        default="http://127.0.0.1:20128/v1",
        alias="LLM_BASE_URL",
    )

    llm_model: str = Field(
        default="gpt-4o-mini",
        alias="LLM_MODEL",
    )

    # Database
    database_url: str = Field(
        default="sqlite:///data/agent.db",
        alias="DATABASE_URL",
    )

    # Canonical relationship: the configured user is permanently Airi's father.
    # Use the numeric Telegram user id (usernames can change).
    # Empty => no configured anchor.
    airi_anchor_user_id: Optional[int] = Field(
        default=None,
        alias="AIRI_ANCHOR_USER_ID",
    )

    # Embedding
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    # Conversation
    max_context_messages: int = Field(
        default=50,
        alias="MAX_CONTEXT_MESSAGES",
    )
    max_memory_results: int = Field(
        default=8,
        alias="MAX_MEMORY_RESULTS",
    )
    memory_importance_threshold: float = Field(
        default=0.25,
        alias="MEMORY_IMPORTANCE_THRESHOLD",
    )

    # Response Timing
    min_response_delay: float = Field(
        default=1.0,
        alias="MIN_RESPONSE_DELAY",
    )
    max_response_delay: float = Field(
        default=8.0,
        alias="MAX_RESPONSE_DELAY",
    )
    direct_mention_delay: float = Field(
        default=0.5,
        alias="DIRECT_MENTION_DELAY",
    )
    autonomous_response_delay: float = Field(
        default=3.0,
        alias="AUTONOMOUS_RESPONSE_DELAY",
    )

    # Autonomous Behavior
    autonomous_behavior_enabled: bool = Field(
        default=True,
        alias="AUTONOMOUS_BEHAVIOR_ENABLED",
    )
    autonomous_check_interval: float = Field(
        default=30.0,
        alias="AUTONOMOUS_CHECK_INTERVAL",
    )
    autonomous_participation_threshold: float = Field(
        default=0.65,
        alias="AUTONOMOUS_PARTICIPATION_THRESHOLD",
    )
    max_initiations_per_hour: int = Field(
        default=3,
        alias="MAX_INITIATIONS_PER_HOUR",
    )

    # Social
    max_messages_per_minute: int = Field(
        default=5,
        alias="MAX_MESSAGES_PER_MINUTE",
    )
    conversation_saturation_threshold: int = Field(
        default=10,
        alias="CONVERSATION_SATURATION_THRESHOLD",
    )
    topic_drift_threshold: float = Field(
        default=0.4,
        alias="TOPIC_DRIFT_THRESHOLD",
    )

    # Working Memory
    working_memory_expiry_minutes: int = Field(
        default=120,
        alias="WORKING_MEMORY_EXPIRY_MINUTES",
    )
    working_memory_max_entries: int = Field(
        default=60,
        alias="WORKING_MEMORY_MAX_ENTRIES",
    )
    working_memory_topic_history: int = Field(
        default=6,
        alias="WORKING_MEMORY_TOPIC_HISTORY",
    )

    # Attention
    attention_direct_mention: float = Field(
        default=0.5,
        alias="ATTENTION_DIRECT_MENTION",
    )
    attention_interest_bonus: float = Field(
        default=0.35,
        alias="ATTENTION_INTEREST_BONUS",
    )
    attention_participation_threshold: float = Field(
        default=0.55,
        alias="ATTENTION_PARTICIPATION_THRESHOLD",
    )

    # Topic Interests
    topic_interest_delta: float = Field(
        default=0.06,
        alias="TOPIC_INTEREST_DELTA",
    )
    topic_interest_max: float = Field(
        default=1.0,
        alias="TOPIC_INTEREST_MAX",
    )
    topic_interest_min: float = Field(
        default=0.0,
        alias="TOPIC_INTEREST_MIN",
    )

    # Boredom
    boredom_threshold: float = Field(
        default=0.6,
        alias="BOREDOM_THRESHOLD",
    )
    boredom_leave_threshold: float = Field(
        default=0.8,
        alias="BOREDOM_LEAVE_THRESHOLD",
    )
    boredom_decay_rate: float = Field(
        default=0.01,
        alias="BOREDOM_DECAY_RATE",
    )

    # Memory Maintenance
    memory_maintenance_interval: int = Field(
        default=900,
        alias="MEMORY_MAINTENANCE_INTERVAL",
    )
    memory_reinforcement_threshold: float = Field(
        default=0.9,
        alias="MEMORY_REINFORCEMENT_THRESHOLD",
    )
    memory_consolidation_interval: int = Field(
        default=3600,
        alias="MEMORY_CONSOLIDATION_INTERVAL",
    )

    # Delayed Actions
    delayed_action_interval: float = Field(
        default=15.0,
        alias="DELAYED_ACTION_INTERVAL",
    )

    # Response Validation
    max_response_chars: int = Field(
        default=400,
        alias="MAX_RESPONSE_CHARS",
    )
    # Casual-chat ceiling: the #1 measured AI tell in her own logs was length
    # (bot median 41 chars vs human median 14). Above this many chars, a
    # message only ships if the room is actually having a long-form moment.
    # 0 disables the gate.
    casual_max_chars: int = Field(
        default=140,
        alias="CASUAL_MAX_CHARS",
    )
    anti_repetition_window: int = Field(
        default=40,
        alias="ANTI_REPETITION_WINDOW",
    )

    # Emotional State
    emotion_decay_rate: float = Field(
        default=0.02,
        alias="EMOTION_DECAY_RATE",
    )
    mood_change_sensitivity: float = Field(
        default=0.15,
        alias="MOOD_CHANGE_SENSITIVITY",
    )

    # Ping fatigue: humans get briefly tired of one person spamming them.
    # Rapid messages within the window stack irritation; time gaps decay it.
    irritation_window_seconds: float = Field(
        default=90.0,
        alias="IRRITATION_WINDOW_SECONDS",
    )
    irritation_increment: float = Field(
        default=0.34,
        alias="IRRITATION_INCREMENT",
    )
    irritation_decay_per_hour: float = Field(
        default=0.6,
        alias="IRRITATION_DECAY_PER_HOUR",
    )
    irritation_max: float = Field(
        default=3.0,
        alias="IRRITATION_MAX",
    )

    # Memory callbacks: resurface one old episodic/relationship memory when
    # it's genuinely relevant. Old = at least this many days; a callback is
    # not re-shown for at least this many days after being surfaced.
    callback_min_age_days: float = Field(
        default=2.0,
        alias="CALLBACK_MIN_AGE_DAYS",
    )
    callback_refractory_days: float = Field(
        default=3.0,
        alias="CALLBACK_REFRACTORY_DAYS",
    )
    # A callback is only offered when retrieval says the memory is genuinely
    # relevant NOW — below this, retelling it would be forced ("nobody asked").
    callback_min_relevance: float = Field(
        default=0.3,
        alias="CALLBACK_MIN_RELEVANCE",
    )

    # Late message self-edits: humans notice their own typo 30s later and
    # silently edit. Chance applies when the immediate-correction dice failed.
    late_typo_fix_chance: float = Field(
        default=0.35,
        alias="LATE_TYPO_FIX_CHANCE",
    )
    late_typo_fix_min_seconds: float = Field(
        default=25.0,
        alias="LATE_TYPO_FIX_MIN_SECONDS",
    )
    late_typo_fix_max_seconds: float = Field(
        default=120.0,
        alias="LATE_TYPO_FIX_MAX_SECONDS",
    )

    # Weekly self-style audit: her own drift (overused openers, question
    # rate, length creep) measured from her recent messages. First run is
    # deliberately soon so the alarms section populates quickly.
    style_audit_interval_seconds: float = Field(
        default=7 * 24 * 3600.0,
        alias="STYLE_AUDIT_INTERVAL_SECONDS",
    )
    style_audit_first_delay_seconds: float = Field(
        default=900.0,
        alias="STYLE_AUDIT_FIRST_DELAY_SECONDS",
    )
    style_audit_window_days: int = Field(
        default=7,
        alias="STYLE_AUDIT_WINDOW_DAYS",
    )

    # Memory
    memory_extraction_interval: int = Field(
        default=180,
        alias="MEMORY_EXTRACTION_INTERVAL",
    )
    max_memories_per_extraction: int = Field(
        default=5,
        alias="MAX_MEMORIES_PER_EXTRACTION",
    )
    memory_similarity_threshold: float = Field(
        default=0.75,
        alias="MEMORY_SIMILARITY_THRESHOLD",
    )

    # Language Behavior
    # Airi speaks English by default; Persian is rare + deliberate.
    default_output_language: str = Field(
        default="english",
        alias="DEFAULT_OUTPUT_LANGUAGE",
    )
    persian_snippet_probability: float = Field(
        default=0.15,
        alias="PERSIAN_SNIPPET_PROBABILITY",
    )
    persian_amusement_threshold: float = Field(
        default=0.75,
        alias="PERSIAN_AMUSEMENT_THRESHOLD",
    )
    persian_affection_threshold: float = Field(
        default=0.4,
        alias="PERSIAN_AFFECTION_THRESHOLD",
    )
    persian_curiosity_threshold: float = Field(
        default=0.8,
        alias="PERSIAN_CURIOSITY_THRESHOLD",
    )

    # Learning Loop / Self-Development
    learning_loop_interval: int = Field(
        default=360,
        alias="LEARNING_LOOP_INTERVAL",
    )
    obsession_engagement_threshold: float = Field(
        default=6.0,
        alias="OBSESSION_ENGAGEMENT_THRESHOLD",
    )
    max_obsessions: int = Field(
        default=5,
        alias="MAX_OBSESSIONS",
    )
    jealousy_decay_rate: float = Field(
        default=0.05,
        alias="JEALOUSY_DECAY_RATE",
    )

    # Conversation Lifecycle
    return_cooldown_seconds: int = Field(
        default=1800,
        alias="RETURN_COOLDOWN_SECONDS",
    )

    # Logging
    debug_mode: bool = Field(
        default=False,
        alias="DEBUG_MODE",
    )
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )
    # "pretty" = colored human-readable console; "json" = machine-readable.
    log_format: str = Field(
        default="pretty",
        alias="LOG_FORMAT",
    )
    verbose_debug: bool = Field(
        default=False,
        alias="VERBOSE_DEBUG",
    )

    # LLM behavior
    # Disable the model's internal "thinking" phase (GLM etc.) — big latency
    # and token savings for her simple JSON tasks. Falls back automatically
    # if the provider rejects the parameter.
    llm_disable_thinking: bool = Field(
        default=True,
        alias="LLM_DISABLE_THINKING",
    )
    # Hard ceiling on one HTTP attempt to the LLM gateway. Without it the
    # OpenAI SDK waits up to 600s per attempt — a hung/slow call holds the
    # per-conversation lock that long and the chat looks frozen.
    llm_timeout_seconds: float = Field(
        default=90.0,
        alias="LLM_TIMEOUT_SECONDS",
    )
    # SDK-internal retries for transient network errors (429/5xx/timeouts).
    # The app has its own retry logic on top; SDK defaults (2) multiply the
    # worst-case latency silently.
    llm_max_retries: int = Field(
        default=1,
        alias="LLM_MAX_RETRIES",
    )

    # Personality dial (state-driven liveliness WITHOUT fine-tuning):
    # LLM sampling temperature is modulated by her emotional state —
    # amused/bored/competitive moments sample hotter, annoyed/embarrassed
    # moments sample cooler (blunt and careful, not random).
    decision_temperature_base: float = Field(
        default=0.5,
        alias="DECISION_TEMPERATURE_BASE",
    )
    decision_temperature_max: float = Field(
        default=0.8,
        alias="DECISION_TEMPERATURE_MAX",
    )

    # Human-imperfection dial (all state-driven, see app/social/style.py):
    # typing register streaks + send-side imperfections. Per-message dice
    # rolls would read as randomness soup; the streaks and probabilities
    # come from her emotional state.
    typing_register_roll_chance: float = Field(
        default=0.10,
        alias="TYPING_REGISTER_ROLL_CHANCE",
    )
    typo_chance_base: float = Field(
        default=0.06,
        alias="TYPO_CHANCE_BASE",
    )
    typo_correction_chance: float = Field(
        default=0.55,
        alias="TYPO_CORRECTION_CHANCE",
    )
    double_text_min_seconds: float = Field(
        default=20.0,
        alias="DOUBLE_TEXT_MIN_SECONDS",
    )
    double_text_max_seconds: float = Field(
        default=45.0,
        alias="DOUBLE_TEXT_MAX_SECONDS",
    )
    # Bleep hard profanity in outgoing messages with [filtered] (streamer
    # style). She still TYPES the actual words — the filter is a platform,
    # not a personality trait. Mild swears (damn/hell/crap) pass through.
    profanity_filter: bool = Field(
        default=True,
        alias="PROFANITY_FILTER",
    )

    # When the deterministic validator hard-blocks a reply (AI filler), she
    # may occasionally answer with a short meta-line instead of vanishing
    # silently ("filtered"). Rate is per-block, so overall frequency stays low.
    filtered_reply_probability: float = Field(
        default=0.25,
        alias="FILTERED_REPLY_PROBABILITY",
    )

    # Persist every reply she sends (with a compact context snapshot) to the
    # training_samples table — the raw material for a future personality
    # fine-tune or prompt-distillation pass. No LLM cost.
    training_sample_logging: bool = Field(
        default=True,
        alias="TRAINING_SAMPLE_LOGGING",
    )

    # Access control. Comma-separated Telegram chat ids. When set, the bot
    # ONLY processes messages/background work in these chats (groups AND
    # private chats); everything else is ignored without any LLM cost.
    # Empty = all chats allowed.
    allowed_chat_ids: str = Field(
        default="",
        alias="ALLOWED_CHAT_IDS",
    )


class PersonalityConfig(BaseModel):
    character: Dict[str, Any]
    personality_traits: Dict[str, float]
    speech_style: Dict[str, Any]
    humor: Dict[str, Any]
    values: list[str]
    dislikes: list[str]
    interests: Dict[str, list[str]]
    behavioral_tendencies: list[str]
    social_tendencies: Dict[str, Any]
    character_boundaries: list[str]
    example_behaviors: list[Dict[str, str]]
    conversation_patterns: Dict[str, str]


def load_personality(
    config_path: Optional[Path] = None,
) -> PersonalityConfig:
    if config_path is None:
        # __file__ = app/config/__init__.py
        # Project root is three levels up.
        config_path = (
            Path(__file__).parent.parent.parent
            / "config"
            / "personality.yaml"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return PersonalityConfig(**data)


_settings = None
_personality = None


def get_allowed_chat_ids() -> Optional[set]:
    """Parse ALLOWED_CHAT_IDS. Returns None when unset (allow all chats),
    otherwise a set of Telegram chat ids (negative for groups)."""
    raw = (get_settings().allowed_chat_ids or "").strip()
    if not raw:
        return None
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            continue
    return ids or None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        try:
            _settings = Settings()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load settings from .env: {e}. "
                "Ensure .env exists with required keys (TELEGRAM_BOT_TOKEN, LLM_API_KEY, etc.)."
            ) from e
    return _settings


def get_personality() -> PersonalityConfig:
    global _personality
    if _personality is None:
        try:
            _personality = load_personality()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load personality.yaml: {e}. "
                "Ensure config/personality.yaml exists and is valid YAML."
            ) from e
    return _personality


settings = get_settings()
personality = get_personality()

from app.config.logging import configure_logging, get_logger  # noqa: E402


__all__ = [
    "Settings",
    "PersonalityConfig",
    "load_personality",
    "settings",
    "personality",
    "configure_logging",
    "get_logger",
]