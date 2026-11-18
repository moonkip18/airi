# Persistent Human-Like Telegram AI Agent

A sophisticated Telegram bot that behaves like a believable human participant in group conversations. Built with a persistent social agent architecture around an LLM, featuring long-term memory, relationships, emotional state, and natural social behavior.

**Status**: 144 tests passing · MIT

## What's actually here (no fake demos)

Most "human-like bot" repos show you a cherry-picked transcript. That proves
nothing — you can't tell a scripted demo from a real one, and neither can I.
So this repo proves its claims the boring way: **unit tests**.

What the 144 tests actually pin down:

- **When she talks and when she doesn't** — the participation scorer decides
  to *ignore* most of what happens in a chat. Tests cover the attention
  weights (direct mention > reply > ambient), conversation saturation, and
  the "a closed exchange stays closed" rule that stops her from reopening a
  dead conversation.
- **What she never says** — a response validator blocks assistant-speak
  ("I understand", "That sounds like", "Hope this helps"), agreement-fishing
  ("right??", "am i right"), and restating the user's message before
  answering. Tested phrase-by-phrase.
- **Memory that behaves like memory** — episodic memories are formed from
  conversations, retrieved by embedding similarity, decay, get superseded
  when contradicted ("passed her exam" → "passed it again"), and *invented*
  memories are rejected. All tested.
- **Human timing** — reply delays are sampled, not instant; direct mentions
  answer faster than ambient join-ins; there's a double-text window and a
  late-typo-fix pass. All tested.
- **Relationship state** — affection/trust/annoyance move per interaction and
  decay over time; irritation at one person spamming accumulates and cools
  off. All tested.

The **LLM is the only untested layer** — it's a pluggable OpenAI-compatible
call (see the model table below), and the whole architecture is built around
not trusting it: structured decisions are Pydantic-validated, prose replies
go through the anti-assistant validator, and anything filtered just doesn't
get sent.

## The eval harness

`python -m tools.day_in_the_life` runs a scripted day (12 scenarios: bait
questions, spam pressure, moments that deserve silence) through the real
pipeline and scores replies on tone/length/freshness/memory rubrics. Without
`--live` it's a free self-test of the scoring machinery (no LLM); with
`--live` it exercises your configured model end-to-end. Treat it as a
regression tool, not a badge.

There's also a **memory-browser dashboard** (FastAPI, localhost) for
inspecting and editing what she remembers —
[`docs/demo-dashboard-api.json`](docs/demo-dashboard-api.json) shows a
captured API response.

## Features

- **Persistent Personality**: Stable character traits defined in YAML, separate from temporary emotions
- **Long-term Memory**: Episodic, semantic, relationship, preference, and conversation summary memories with embeddings
- **Semantic Memory Retrieval**: Vector-based retrieval with reranking by relevance, importance, recency, and relationship
- **Relationship Tracking**: Affection, trust, respect, annoyance, familiarity, interest per user
- **Emotional State**: Mood, energy, curiosity, amusement, annoyance, interest, social engagement
- **Conversation Lifecycle**: Away → Observing → Interested → Participating → Winding Down → Away
- **Autonomous Participation**: Joins conversations without being mentioned, based on social appropriateness
- **Natural Timing**: Realistic response delays based on context
- **Structured LLM Decisions**: Pydantic-validated social actions (reply, ignore, react, initiate, leave)
- **Memory Formation**: LLM-powered extraction of meaningful memories from conversations
- **Contradiction Handling**: Supersedes outdated memories with temporal validity
- **Topic Drift Prevention**: Stays on current conversation topic unless genuine reason to shift

## Architecture

```
Telegram
    ↓
Message Ingestion
    ↓
Conversation / Social Perception
    ↓
Current Conversation State
    ↓
Memory Retrieval (semantic + reranking)
    ↓
Personality + Relationship State + Emotional State
    ↓
LLM Reasoning (structured output)
    ↓
Social Decision
    ↓
Action Execution
    ↓
Memory/State Updates
```

## Quick Start

### Prerequisites

- Python 3.11+
- Groq API key
- Telegram bot token (from @BotFather)

### Installation

```bash
git clone <repo>
cd project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables:
```env
TELEGRAM_BOT_TOKEN=your_bot_token
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

Any OpenAI-compatible endpoint works — OpenAI, Groq, OpenRouter, or a local
Ollama/LM Studio server. Recommended models (good at casual, personality-
driven chat, cheap enough to run all day):

| Provider | Model | `LLM_BASE_URL` |
|---|---|---|
| OpenAI | `gpt-4o-mini` | `https://api.openai.com/v1` |
| Groq | `llama-3.3-70b-versatile` | `https://api.groq.com/openai/v1` |
| OpenRouter | `meta-llama/llama-3.3-70b-instruct` | `https://openrouter.ai/api/v1` |
| Local (Ollama) | `qwen2.5:14b` | `http://127.0.0.1:11434/v1` |

Larger models work but are rarely worth it for casual chat; sub-7B models
tend to slip into assistant-speak, which the validator filters at a cost.

Optional configuration (see `.env.example` for all options):
```env
MAX_CONTEXT_MESSAGES=30
MAX_MEMORY_RESULTS=5
AUTONOMOUS_BEHAVIOR_ENABLED=true
DEBUG_MODE=false
```

### Running

```bash
python main.py
```

The bot will start polling for messages. Add it to a group or message it directly.

## Personality Configuration

Edit `config/personality.yaml` to customize the character:

```yaml
character:
  name: "Alex"
  age: "mid-20s"
  background: "software developer..."

personality_traits:
  openness: 0.75
  conscientiousness: 0.55
  extraversion: 0.45
  agreeableness: 0.65
  neuroticism: 0.35

speech_style:
  formality: 0.3
  verbosity: 0.5
  humor: 0.7
  sarcasm: 0.6
  emoji_usage: 0.4
```

The personality defines **permanent traits** only. Temporary states (mood, relationships, memories) are managed separately.

## Memory System

### Memory Types

| Type | Description | Example |
|------|-------------|---------|
| Episodic | Specific events | "Alex passed physics exam yesterday" |
| Semantic | Generalized knowledge | "Alex dislikes chemistry" |
| Relationship | Dynamics with a person | "Trust Alex with technical advice" |
| Preference | Stable likes/dislikes | "Prefers dark mode editors" |
| Conversation Summary | Compressed history | "Discussed Rust vs Go for 20 mins" |

### Memory Fields

```python
id, timestamp, type, content, people_involved, topics,
importance, confidence, source, embedding, last_retrieved,
superseded_by, valid_from, valid_until
```

### Retrieval Pipeline

1. **Query Construction**: Combines current message, topic, participants, emotional state
2. **Candidate Retrieval**: Semantic similarity search (top 3x limit)
3. **Reranking**: Weighted scoring
   - Semantic similarity: 45%
   - Importance: 20%
   - Recency: 15%
   - Relationship relevance: 15%
   - Redundancy penalty: -10%
4. **Deduplication**: Removes topically similar memories
5. **Return**: Top N highly relevant memories

### Contradiction Handling

When new information contradicts old memories:
- Old memory gets `superseded_by` pointer to new memory
- Old memory confidence reduced
- Temporal validity (`valid_from`/`valid_until`) allows both to coexist
- Example: "Alex disliked physics in 2025" + "Alex currently enjoys physics"

## Social Decision System

### Participation Factors

| Factor | Weight | Description |
|--------|--------|-------------|
| Direct mention | 35% | Bot was @mentioned or replied to |
| Topic relevance | 20% | Message relates to current topic |
| Personal interest | 15% | Matches bot's interests/memories |
| Relationship relevance | 10% | Affection/trust with sender |
| Conversation energy | 5% | How active the chat is |
| Curiosity | 5% | Bot's current curiosity level |
| Current mood | 5% | Positive mood increases participation |
| Recent activity | -5% | Penalty for speaking too much |
| Saturation | -5% | Penalty if bot has spoken a lot recently |

### Decision Thresholds

- **Direct mention**: Score > 0.3
- **Normal**: Score > 0.4
- **Autonomous**: Score > 0.65 (configurable)

### Actions

```python
class SocialAction(Enum):
    REPLY = "reply"      # Respond to current message
    IGNORE = "ignore"    # Stay silent
    REACT = "react"      # Emoji reaction only
    INITIATE = "initiate" # Start new topic
    LEAVE = "leave"      # Exit conversation
```

## Conversation Lifecycle

```
AWAY (not aware)
  ↓ conversation becomes interesting
OBSERVING (watching, not participating)
  ↓ decides to join
INTERESTED → PARTICIPATING (actively chatting)
  ↓ conversation winds down
WINDING_DOWN (signaling departure)
  ↓
AWAY
```

The bot can say things like "alright I'm gonna go" and won't immediately respond to subsequent messages.

## Autonomous Behavior

Background loop runs every 30s (configurable):
1. Scans active group conversations
2. Evaluates participation score for recent messages
3. If score > threshold and not rate-limited → initiates response
4. Uses longer delays (3s+ base) for autonomous responses

## Response Timing

| Context | Base Delay | Variation |
|---------|-----------|-----------|
| Direct mention | 0.5s | ±0.5s |
| Autonomous | 3.0s | ±1.5s |
| Normal reply | 1.0s | ±1.5s |

Complex messages and fast conversations increase delay. Max delay: 8s.

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Memory storage, retrieval, and contradiction handling
- Conversation state transitions
- Emotional state updates
- Relationship dynamics
- Social decision scoring
- Agent message processing
- Personality loading

## Project Structure

```
project/
├── app/
│   ├── agent/           # Core agent logic, context building, background tasks
│   ├── config/          # Settings, personality loader, logging
│   ├── database/        # SQLAlchemy models, session management
│   ├── llm/             # LLM provider interface + Groq implementation
│   ├── memory/          # Embeddings, storage, retrieval, formation
│   ├── personality/     # Personality dataclasses and loader
│   ├── social/          # Decision engine, participation scoring
│   ├── state/           # Conversation, emotional, relationship managers
│   └── telegram/        # Bot handler, message processing
├── config/
│   └── personality.yaml # Character definition
├── data/                # SQLite database (created at runtime)
├── tests/               # Automated tests
├── .env.example         # Configuration template
├── requirements.txt     # Dependencies
└── main.py              # Entry point
```

## Replacing Components

### LLM Provider

Implement `LLMProvider` interface in `app/llm/`:

```python
class MyProvider(LLMProvider):
    async def generate(self, messages, temperature, max_tokens, response_format):
        ...

    async def generate_structured(self, messages, response_model, temperature, max_tokens):
        ...

    async def close(self):
        ...
```

Update `app/llm/__init__.py` to export your provider.

### Embedding Model

Replace `EmbeddingModel` in `app/memory/embeddings.py`:

```python
class MyEmbeddings:
    def encode(self, texts): ...
    def encode_single(self, text): ...
    def similarity(self, a, b): ...
```

### Database

Uses SQLAlchemy - change `DATABASE_URL` to PostgreSQL/MySQL. Models in `app/database/models.py`.

### Telegram Interface

The bot logic is in `app/telegram/bot.py`. Could be adapted for Discord, Slack, etc.

## Observability

Structured JSON logs (or pretty console in debug mode) include:

```json
{
  "event": "social_decision_made",
  "action": "reply",
  "confidence": 0.82,
  "reasoning": "Direct mention, high topic relevance",
  "participation_score": 0.73
}
```

Key log events:
- `processing_message` - Incoming message details
- `participation_decision` - Score and threshold decision
- `memory_retrieval_complete` - Candidates, returned, top score
- `memory_extracted` - New memories formed
- `emotional_state_updated` - Event type, new state values
- `relationship_updated` - User, interaction type, new affection
- `conversation_state_changed` - Old/new state

## Design Principles

1. **LLM for reasoning/language, Python for identity persistence**
2. **Separate permanent personality from temporary state**
3. **Quality over quantity in memory** - only meaningful events stored
4. **Contextual relevance > semantic similarity** for memory retrieval
5. **Silence is a valid social action**
6. **Conversations have natural lifecycles**
7. **Relationships modulate but don't dictate behavior**
8. **Structured LLM outputs enable controllable behavior**

## License

MIT