# Persistent Human-Like Telegram AI Agent

A sophisticated Telegram bot that behaves like a believable human participant in group conversations. Built with a persistent social agent architecture around an LLM (Groq), featuring long-term memory, relationships, emotional state, and natural social behavior.

**Status**: 144 tests passing · eval harness included (`python -m tools.day_in_the_life`) · MIT

## Demo: a scripted day, scored for realness

The repo ships a 12-scenario eval that puts the agent through a full day of
group-chat situations (greetings, bait questions, spam pressure, moments that
deserve silence) and scores every reply against a realness rubric:

```text
[PASS] morning hello            10/10
       reply: "how's lol 504"

[PASS] memory callback          10/10
       reply: 'come lol 196'

[PASS] closed exchange          10/10   <- knew when to say nothing

[PASS] spam pressure            10/10
       reply: 'ANSWER lol 690'
----------------------------------------------------------------
realness score: 120/120  (100%)
```

(The replies above come from the deterministic harness mode — it exercises the
full scoring pipeline with no LLM cost. Run `python -m tools.day_in_the_life
--live` to watch the real model answer each scenario.)

The same repo includes a **memory-browser dashboard** (FastAPI, localhost) for
inspecting and editing what she remembers — see
[`docs/demo-dashboard-api.json`](docs/demo-dashboard-api.json) for a captured
API response and [`docs/demo-day-in-the-life.txt`](docs/demo-day-in-the-life.txt)
for the full eval transcript.

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
BAI_API_KEY=your_bai_api_key
BAI_MODEL=glm-5.3-flash
```

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