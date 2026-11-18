# agent.md

Guidance for AI coding agents (and humans) working in this repository.

## What this is

**Airi** — a persistent, human-like Telegram AI agent. A Python asyncio service that
participates in group/private chats like a believable person: it has a stable
personality, long-term semantic memory, per-user relationships, an emotional state,
a conversation lifecycle, and an LLM-driven social decision system.

## Commands

```bash
python main.py            # run the bot (polling)
pytest tests/ -v          # run the test suite
pip install -r requirements.txt
```

- Python **3.11+**. Everything is async (`asyncio`, SQLAlchemy 2 async, PTB 21+).
- Config lives in `.env` (template: `.env.example`). Never commit `.env`.
- The character is defined in `config/personality.yaml` — permanent traits only;
  temporary state (mood, relationships, memories) lives in the database.

## Layout

```
main.py                Entry point: init_db → TelegramBot → BackgroundTasks → run
app/
  agent/               Core pipeline (core.py), context building (context.py),
                       background loops (background.py), self-model/learning (learning.py)
  analysis/            Cheap, non-LLM message analysis + reply-target classification
  config/              Pydantic Settings (.env), personality loader, structlog setup
  database/            SQLAlchemy models, async session factory, SQLite migrations
  language/            Input language detection + output language policy (EN default,
                       rare deliberate Persian snippets)
   llm/                 LLMProvider interface (base.py); RouterProvider (local
                        9router, OpenAI-compatible) in router.py, exported as
                        `llm_provider`
  memory/              Embeddings (sentence-transformers w/ local fallback), memory store
                       & formation, semantic retrieval + reranking, maintenance (decay/
                       reinforcement/consolidation)
  personality/         Personality dataclasses + YAML loader (`get_personality()`)
  social/              Decision engine, participation & attention scoring, response
                       validation / anti-repetition, delayed action scheduler
  state/               Conversation lifecycle, emotional state, relationships,
                       working memory, topic interests, social graph
  telegram/            PTB handlers, keyboards, proxy wiring, concurrency config
config/personality.yaml  Character definition
data/                  SQLite database (runtime)
tests/                 pytest suite (see conftest.py for fixtures)
tools/                 rate_samples.py — curate training_samples (distillation
                       loop: rated samples become voice exemplars in prompts)
```

## Critical invariants — do not break

1. **Shared `SocialDecisionEngine`**: the engine is created once in `TelegramBot` and
   passed into `Agent`. Rate limiting / burst counters live on it; never instantiate
   a fresh engine per message.
2. **SQLite write discipline**: writes go through `db_write_lock`
   (`app/database/session.py`) — an `asyncio.Lock` shared by handlers and background
   loops. WAL mode + 30s busy_timeout are configured in `init_db()`. Do not add raw
   writes outside this pattern.
3. **Schema changes**: SQLite can't alter existing tables. Add new columns to
   `_COLUMN_MIGRATIONS` in `app/database/session.py` (ALTER TABLE runs on startup)
   **and** to the model in `app/database/models.py`.
4. **One `LLMProvider` global**: `llm_provider` from `app.llm`. LLM calls are
   expensive — batch where possible (e.g. memory extraction is one call per window,
   learning is one call per conversation window, never per message).
5. **Cost-gated background loops**: extraction/learning loops check how many NEW
   messages arrived since the last run before spending tokens. The gateway token
   budget is generous (30M) — gates are loose and quality-first; they exist to
   skip silent chats, NOT to ration normal activity. Social frequency gates
   (rate limits, initiation cooldowns) are a realism concern and stay strict.
6. **Access control**: when `ALLOWED_CHAT_IDS` is set, every handler and every
   background loop must ignore other chats (zero API cost). Keep the `_chat_filter`
   behavior intact in `background.py`.
7. **Reply chains**: `Message.reply_to_user_id` must always be populated with the
   internal user id of the original sender — reply context must never be lost.
8. **Canonical relationships**: `AIRI_ANCHOR_USER_ID` sets a canonical
   `relationship_type` protected by `is_canonical` — never overwritten by learned
   interaction data, never decayed.

## Conventions

- **Comments explain "why"**, not "what" — follow the existing style (rationale,
  edge cases, failure history).
- **Logging**: `structlog` with snake_case event names (`"memory_retrieval_complete"`,
  `"social_decision_made"`). Add structured key/value context, not prose.
- **Config**: all tunables go in `Settings` (pydantic-settings) with an alias matching
  the env var, and a corresponding entry in `.env.example`.
- **Pydantic models** for all structured LLM output; validated via
  `generate_structured(...)`.
- **Enums** for closed sets: `SocialAction` (REPLY/IGNORE/REACT/INITIATE/LEAVE/WAIT),
  `ConversationState` (AWAY → OBSERVING → INTERESTED → PARTICIPATING → WINDING_DOWN),
  `MemoryType` (episodic, semantic, relationship, preference, conversation_summary,
  behavioral_pattern).
- **Silence is a valid decision** — never force the bot to reply; IGNORE/WAIT are
  first-class outcomes of the decision pipeline.
- **Windows support**: `main.py` avoids `add_signal_handler` on win32; keep
  platform checks when touching lifecycle code. Proxy support (`TELEGRAM_PROXY`)
  must stay wired into both HTTPX request pools.
- Response timing delays, participation thresholds, etc. come from settings — do not
  hardcode magic numbers that already exist in `.env.example`.

## Testing

- `pytest tests/ -v` — tests cover memory storage/retrieval/contradictions, state
  transitions, emotional updates, relationships, decision scoring, agent processing,
  personality loading, and past audit fixes.
- New modules should come with tests; reuse fixtures in `tests/conftest.py`.
- Never write tests that hit the real Telegram/LLM gateway APIs; mock the LLM
  provider and Telegram objects.

## Common pitfalls

- README.md's "Groq" references are historical; the active provider is
  **the local 9router (OpenAI-compatible gateway)** — see `app/llm/router.py` and `.env.example`.
- `concurrent_updates(256)` + per-conversation locks in `bot.py`: bursts are processed
  concurrently but same-chat ordering is preserved. Don't serialize everything.
- `nullpool` + async engine: sessions are per-task; don't share `AsyncSession`
  across tasks.
- Embeddings fall back to a light local hashed backend if sentence-transformers/torch
  is unavailable — code must work under both paths.
