# architecture.md

Technical architecture of **Airi**, a persistent human-like Telegram AI agent.

## 1. Overview

Airi is an async Python service that ingests Telegram messages, builds social
perception from cheap heuristics + persistent state, retrieves relevant memories,
asks an LLM for a **structured social decision**, executes it with human-like
timing, and then updates memory/emotion/relationships — all persisted in SQLite.

Design principle: **the LLM handles reasoning and language; Python owns identity
persistence** (personality, memory, relationships, emotions). LLM weights are never
modified; "learning" happens through persistent state.

## 2. System flow

```
Telegram (long polling, concurrent_updates=256)
    ↓
TelegramBot (app/telegram/bot.py)         per-conversation asyncio locks
    ↓                                       shared SocialDecisionEngine
Agent.process_message (app/agent/core.py)
    ↓
┌─ STEP 0  Message analysis + targeting (app/analysis)        — no LLM
├─ STEP 1  Language detection (app/language)                  — no LLM
├─ STEP 2  State managers: conversation lifecycle, emotional
│          state, relationships, working memory, topic
│          interests, social graph (app/state)
├─ STEP 3  Memory retrieval: embeddings → semantic search →
│          rerank (similarity/importance/recency/relationship
│          /redundancy) → dedupe (app/memory)
├─ STEP 4  Context building (app/agent/context.py)
├─ STEP 5  LLM structured decision → SocialAction:
│          REPLY | IGNORE | REACT | INITIATE | LEAVE | WAIT
│          (app/llm + app/social/decision.py)
├─ STEP 6  Response validation + anti-repetition
│          (app/social/validation.py)
└─ STEP 7  Delayed, human-like send (app/social/delayed.py)
    ↓
Memory formation / state updates (persisted)
```

## 3. Process & lifecycle (`main.py`)

`Application` orchestrates ordered startup and teardown:

1. `init_db()` — create tables, apply column migrations, WAL pragmas
2. `TelegramBot(token)` — PTB `Application` with proxy-aware HTTPX pools
3. `BackgroundTasks(bot)` — 8 async loops (see §6)
4. `bot.start()` — polling; shutdown event (POSIX signals / KeyboardInterrupt)

Shutdown order: background tasks → bot → database (last).

## 4. Module map

| Module | Responsibility |
|---|---|
| `app/agent/core.py` | `Agent` — the message pipeline; composes all managers |
| `app/agent/context.py` | `ContextBuilder` — LLM prompt context (history, memories, state) |
| `app/agent/background.py` | `BackgroundTasks` — periodic loops |
| `app/agent/learning.py` | `SelfModelManager` / `LearningLoop` — obsessions, jokes, tendencies, topic engagement |
| `app/analysis/` | `analyze_message` (language, spam/repetition hashes, questions…), `classify_target` (who a message addresses) |
| `app/config/` | `Settings` (pydantic-settings, `.env`), personality loader, structlog config |
| `app/database/` | Models + `get_session()` + `db_write_lock` + startup migrations |
| `app/language/` | Input detection, output policy (EN default; rare deliberate Persian snippets gated by emotion/relationship), per-user language stats |
| `app/llm/` | `LLMProvider` ABC; `RouterProvider` (local 9router, OpenAI-compatible, `router.py`) as `llm_provider` |
| `app/memory/` | Embeddings (MiniLM; local hashed fallback), store/formation, retrieval + reranking, maintenance |
| `app/personality/` | Permanent traits from `config/personality.yaml` |
| `app/social/` | Decision engine + scorer, attention scorer, response validator / anti-repetition, delayed action scheduler |
| `app/state/` | Conversation lifecycle, emotional state, relationships, working memory, topic interests, social graph |
| `app/telegram/` | PTB wiring: handlers, concurrency, proxy, keyboards |

## 5. Data model (`app/database/models.py`)

Core tables:

- `users` — Telegram users (internal id ↔ telegram_id)
- `conversations` / `conversation_participants` — chats and membership
- `messages` — full history incl. `reply_to_user_id` (reply chains never lose the
  original sender) and `is_from_bot`
- `memories` — typed memories with `embedding` (JSON floats), `importance`,
  `confidence`, `reinforcement`, `last_retrieved`, and temporal contradiction
  fields (`superseded_by`, `valid_from`, `valid_until`)
- `relationships` — per-user: affection, trust, respect, annoyance, familiarity,
  interest, comfort, jealousy; `relationship_type` with `is_canonical` protection
- `emotional_state` — mood, energy, curiosity, amusement, annoyance, interest,
  embarrassment, competitiveness, boredom, …
- `conversation_states` — lifecycle state per chat, recent topics, pending topic,
  initiation counters
- `conversation_summaries` — compressed history
- `recent_phrases` — anti-repetition window
- `self_model` — obsessions, recurring jokes, learned tendencies, topic engagement

Enums: `MemoryType` (episodic, semantic, relationship, preference,
conversation_summary, behavioral_pattern), `ConversationState`
(AWAY → OBSERVING → INTERESTED → PARTICIPATING → WINDING_DOWN), `ActionType`.

### SQLite strategy (`app/database/session.py`)

- async engine (`aiosqlite`), `NullPool`, WAL, `busy_timeout=30000`, FKs on
- **`db_write_lock`** — a global `asyncio.Lock`; handlers and background loops
  serialize writes through it
- **`_COLUMN_MIGRATIONS`** — lightweight startup migrations: any missing column is
  added via `ALTER TABLE` before `create_all` (SQLite cannot alter existing tables)
- `get_session()` — context manager that commits on success, rolls back on error

## 6. Background loops (`app/agent/background.py`)

| Loop | Interval (default) | Purpose |
|---|---|---|
| Emotional decay + circadian | 60s | decay emotions; pull energy/social toward time-of-day baseline |
| Memory extraction | 300s | LLM extracts memories from new messages (**cost-gated**: runs only if new messages exist) |
| Learning loop | 600s | one batched LLM call per window → self-model / relationship / emotion deltas |
| Conversation state | periodic | lifecycle transitions (interest → participate → wind down) |
| Autonomous scan | 30s | score recent messages; initiate when > threshold and rate-limits allow |
| Memory maintenance | 900s | decay / reinforcement / consolidation of memories |
| Delayed actions | 15s | execute scheduled (human-like delayed) replies/reactions |
| Conversation summary | 3600s | compress old history into summary memories |

All loops respect `ALLOWED_CHAT_IDS` access control via `_chat_filter` and wrap
DB work in `db_write_lock`.

## 7. Social decision system

Two scoring layers feed the LLM decision:

1. **Attention scoring** (`social/attention.py`) — direct mentions, interest bonus,
   participation threshold → does the bot even consider the message?
2. **Participation scoring** (`social/decision.py`) — weighted factors
   (mention 35%, topic relevance 20%, personal interest 15%, relationship 10%,
   energy/curiosity/mood 5% each, recent activity & saturation penalties).
   Thresholds: mention > 0.3, normal > 0.4, autonomous > 0.65.

The LLM then chooses a structured `SocialAction` (REPLY / IGNORE / REACT /
INITIATE / LEAVE / WAIT) with reasoning; outputs are Pydantic-validated.
**Silence is a first-class outcome.**

Response timing is contextual (mention 0.5s, autonomous 3s, normal 1s; complex
messages and busy chats add delay; capped by `MAX_RESPONSE_DELAY=8s`).

## 8. Memory pipeline

- **Formation**: LLM extraction (batched, max 3 per window, importance threshold)
- **Storage**: embedding computed via `all-MiniLM-L6-v2` (sentence-transformers),
  or a light local hashed backend fallback
- **Retrieval**: query construction (message + topic + participants + emotion) →
  top-3x candidate semantic search → rerank
  (similarity 45% / importance 20% / recency 15% / relationship 15% / redundancy −10%)
  → topical dedupe → top-N
- **Contradictions**: old memory gets `superseded_by` + reduced confidence;
  temporal validity allows "disliked X in 2025" and "enjoys X now" to coexist
- **Maintenance**: periodic decay, reinforcement of reconfirmed facts, consolidation

## 9. Language behavior

- Input: heuristic Persian/English detection (`language/detector.py`)
- Output: policy (`language/policy.py`) — English by default; occasional deliberate
  Persian snippets gated by amusement/affection/curiosity thresholds and per-user
  stats (`language/stats.py`)

## 10. LLM layer

- `LLMProvider` ABC: `generate`, `generate_structured` (Pydantic models), `close`
- Active provider: `RouterProvider` (local 9router, OpenAI-compatible, `LLM_*`
  settings); `LLM_DISABLE_THINKING=true` saves tokens/latency
- Swap providers by implementing the ABC and re-exporting in `app/llm/__init__.py`

## 11. Telegram layer

- PTB `Application` with `concurrent_updates(256)` for throughput; per-conversation
  `asyncio.Lock` preserves same-chat ordering; `_pending_replies` avoids duplicates
- Proxy (`TELEGRAM_PROXY`) wired into both request pools (socks5 supported)
- Non-text content (stickers/photos/GIFs/voice) is perceived as short descriptions

## 12. Observability

structlog JSON logs (pretty in debug). Key events: `processing_message`,
`participation_decision`, `social_decision_made`, `memory_retrieval_complete`,
`memory_extracted`, `emotional_state_updated`, `relationship_updated`,
`conversation_state_changed`, `chat_ignored_not_allowed`.

## 13. Extension points

| Replace | How |
|---|---|
| LLM provider | implement `LLMProvider` (`app/llm/base.py`), export in `app/llm/__init__.py` |
| Embeddings | replace `EmbeddingModel` (`app/memory/embeddings.py`) |
| Database | set `DATABASE_URL` (SQLAlchemy async; models in `app/database/models.py`) |
| Chat platform | reimplement `app/telegram/bot.py` glue — the agent core is platform-agnostic |

## 14. Configuration

All tunables are in `.env` (see `.env.example`): tokens, database, memory limits,
timing, autonomy thresholds, boredom, language, learning loop, access control
(`ALLOWED_CHAT_IDS`), logging, and `AIRI_ANCHOR_USER_ID` (canonical relationship).
