"""Reaction perception tests.

She sees reactions to messages (hers and others'), stores them as
pseudo-message rows so they enter context and long-term memory, gets a
small emotional/relationship nudge when HER messages are reacted to —
and never replies to any of them.
"""
import pytest
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from unittest.mock import patch

from telegram import (
    Chat,
    MessageReactionUpdated,
    ReactionTypeEmoji,
    Update,
    User as TgUser,
)

from app.database import Message, User
from app.memory.store import get_or_create_user, get_or_create_conversation
from app.state import EmotionalStateManager, RelationshipManager
from app.telegram.bot import TelegramBot, _recent_reaction_nudges

from sqlalchemy import select


BOT_TG_ID = 777000000


@pytest.fixture(autouse=True)
def _reset_reaction_state():
    _recent_reaction_nudges.clear()
    yield
    _recent_reaction_nudges.clear()


@pytest.fixture
def bot():
    return TelegramBot("123456:FAKE")


@pytest.fixture
def session_factory_patch(monkeypatch, session):
    """Route handle_reaction's `async with get_session()` to the test DB."""
    @asynccontextmanager
    async def _fake_get_session():
        yield session

    monkeypatch.setattr(
        "app.telegram.bot.get_session", lambda: _fake_get_session()
    )
    return session


@pytest.fixture
def allow_all_chats(monkeypatch):
    monkeypatch.setattr("app.config.get_allowed_chat_ids", lambda: None)


def _reaction_update(
    actor_tg_id: int,
    chat_id: int = -100999,
    message_id: int = 100,
    emoji="❤",
    username="rae",
    first_name="Rae",
    is_bot: bool = False,
    new_reaction=None,
):
    if new_reaction is None:
        new_reaction = (ReactionTypeEmoji(emoji=emoji),) if emoji else ()

    return Update(
        update_id=1,
        message_reaction=MessageReactionUpdated(
            chat=Chat(id=chat_id, type="supergroup", title="Test Chat"),
            message_id=message_id,
            date=datetime.utcnow(),
            old_reaction=(),
            new_reaction=new_reaction,
            user=TgUser(
                id=actor_tg_id,
                is_bot=is_bot,
                first_name=first_name,
                username=username,
            ),
        ),
    )


async def _seed_her_message(session, conv, text="lol yeah", tg_id=100):
    bot_user_id = await TelegramBot._get_bot_user_id(
        TelegramBot("123456:FAKE"), session
    )
    msg = Message(
        conversation_id=conv.id,
        user_id=bot_user_id,
        telegram_message_id=tg_id,
        text=text,
        timestamp=datetime.utcnow(),
        is_from_bot=True,
    )
    session.add(msg)
    await session.commit()
    return bot_user_id


@pytest.mark.asyncio
async def test_reaction_to_her_message_recorded_with_reply_chain(
    session, session_factory_patch, allow_all_chats, bot
):
    conv = await get_or_create_conversation(session, -100999, "supergroup")
    her_id = await _seed_her_message(session, conv)
    actor = await get_or_create_user(session, 555, "rae", "Rae", None)
    await session.commit()

    await bot.handle_reaction(
        _reaction_update(555, message_id=100, emoji="❤"), None
    )

    rows = (
        await session.execute(
            select(Message).where(Message.user_id == actor.id)
        )
    ).scalars().all()
    reaction_rows = [r for r in rows if (r.text or "").startswith("[reacted")]

    assert len(reaction_rows) == 1
    row = reaction_rows[0]
    assert row.text == "[reacted ❤ to your message]"
    # Reply chain invariant: points at HER internal id, so the agent's
    # "am I being ignored?" check counts the reaction as engagement.
    assert row.reply_to_user_id == her_id
    assert row.is_from_bot is False
    assert row.conversation_id == conv.id

    # Emotional nudge: mood rises, never a mood HIT.
    emo = await EmotionalStateManager(session).get_state()
    assert emo["mood"] > 0

    # Relationship nudge: positive interaction with the reactor.
    rel = await RelationshipManager(session).get_relationship(actor.id)
    assert rel is not None
    assert rel.interaction_count >= 1
    assert rel.affection > 0


@pytest.mark.asyncio
async def test_reaction_to_other_users_message_recorded_without_nudge(
    session, session_factory_patch, allow_all_chats, bot
):
    conv = await get_or_create_conversation(session, -100999, "supergroup")
    other = await get_or_create_user(session, 42, "zara", "Zara", None)
    session.add(Message(
        conversation_id=conv.id,
        user_id=other.id,
        telegram_message_id=200,
        text="hello everyone",
        timestamp=datetime.utcnow(),
        is_from_bot=False,
    ))
    actor = await get_or_create_user(session, 555, "rae", "Rae", None)
    await session.commit()

    await bot.handle_reaction(
        _reaction_update(555, message_id=200, emoji="👀"), None
    )

    rows = (
        await session.execute(
            select(Message).where(Message.user_id == actor.id)
        )
    ).scalars().all()
    reaction_rows = [r for r in rows if (r.text or "").startswith("[reacted")]

    assert len(reaction_rows) == 1
    row = reaction_rows[0]
    assert "Zara" in row.text or "zara" in row.text
    assert row.reply_to_user_id == other.id

    # Reactions to OTHERS' messages carry no emotional nudge for her.
    emo = await EmotionalStateManager(session).get_state()
    assert emo["mood"] == 0.0


@pytest.mark.asyncio
async def test_reaction_to_untracked_message_recorded_anonymously(
    session, session_factory_patch, allow_all_chats, bot
):
    conv = await get_or_create_conversation(session, -100999, "supergroup")
    actor = await get_or_create_user(session, 555, "rae", "Rae", None)
    await session.commit()

    await bot.handle_reaction(
        _reaction_update(555, message_id=999, emoji="🔥"), None
    )

    rows = (
        await session.execute(
            select(Message).where(Message.user_id == actor.id)
        )
    ).scalars().all()
    reaction_rows = [r for r in rows if (r.text or "").startswith("[reacted")]

    assert len(reaction_rows) == 1
    assert reaction_rows[0].text == "[reacted 🔥]"
    assert reaction_rows[0].reply_to_user_id is None


@pytest.mark.asyncio
async def test_reaction_removal_not_recorded(
    session, session_factory_patch, allow_all_chats, bot
):
    conv = await get_or_create_conversation(session, -100999, "supergroup")
    await _seed_her_message(session, conv)
    actor = await get_or_create_user(session, 555, "rae", "Rae", None)
    await session.commit()

    await bot.handle_reaction(
        _reaction_update(555, message_id=100, new_reaction=()), None
    )

    rows = (
        await session.execute(
            select(Message).where(Message.user_id == actor.id)
        )
    ).scalars().all()
    assert not [r for r in rows if (r.text or "").startswith("[reacted")]


@pytest.mark.asyncio
async def test_own_reaction_ignored(
    session, session_factory_patch, allow_all_chats, bot
):
    conv = await get_or_create_conversation(session, -100999, "supergroup")
    await _seed_her_message(session, conv)
    before = (
        await session.execute(select(Message))
    ).scalars().all()

    # initialize() never ran, so set the id the guard compares against.
    bot.bot_id = BOT_TG_ID

    # Her own reaction: bot flag set AND bot id match — both guards.
    await bot.handle_reaction(
        _reaction_update(BOT_TG_ID, message_id=100, emoji="❤", is_bot=True),
        None,
    )
    await bot.handle_reaction(
        _reaction_update(BOT_TG_ID, message_id=100, emoji="❤"), None
    )

    after = (
        await session.execute(select(Message))
    ).scalars().all()
    assert len(after) == len(before)


@pytest.mark.asyncio
async def test_reaction_in_not_allowed_chat_ignored(
    session, session_factory_patch, monkeypatch, bot
):
    monkeypatch.setattr(
        "app.config.get_allowed_chat_ids", lambda: {-100999}
    )
    conv = await get_or_create_conversation(session, -100999, "supergroup")
    await _seed_her_message(session, conv)
    await session.commit()
    before = (
        await session.execute(select(Message))
    ).scalars().all()

    # Chat not in the allowlist -> zero cost, nothing stored.
    await bot.handle_reaction(
        _reaction_update(555, chat_id=-111, message_id=100), None
    )

    after = (
        await session.execute(select(Message))
    ).scalars().all()
    assert len(after) == len(before)


@pytest.mark.asyncio
async def test_nudge_cooldown_limits_affection_farming(
    session, session_factory_patch, allow_all_chats, bot
):
    conv = await get_or_create_conversation(session, -100999, "supergroup")
    await _seed_her_message(session, conv)
    actor = await get_or_create_user(session, 555, "rae", "Rae", None)
    await session.commit()

    await bot.handle_reaction(_reaction_update(555, message_id=100), None)
    first_rel = await RelationshipManager(session).get_relationship(actor.id)
    first_count = first_rel.interaction_count

    # Rapid re-reacts within the cooldown must not farm relationship nudge.
    for msg_id in (100, 100):
        await bot.handle_reaction(
            _reaction_update(555, message_id=msg_id), None
        )

    second_rel = await RelationshipManager(session).get_relationship(actor.id)
    assert second_rel.interaction_count == first_count


@pytest.mark.asyncio
async def test_reaction_received_emotional_event(session):
    manager = EmotionalStateManager(session)

    before = await manager.get_state()
    await manager.update_from_event("reaction_received", 1.0)
    after = await manager.get_state()

    assert after["mood"] > before["mood"]
    assert after["amusement"] > before["amusement"]
    assert after["boredom"] < before["boredom"]


@pytest.mark.asyncio
async def test_reaction_row_reaches_working_memory_window(
    session, session_factory_patch, allow_all_chats, bot
):
    conv = await get_or_create_conversation(session, -100999, "supergroup")
    await _seed_her_message(session, conv)
    await get_or_create_user(session, 555, "rae", "Rae", None)
    await session.commit()

    await bot.handle_reaction(
        _reaction_update(555, message_id=100, emoji="🥰"), None
    )

    # The pseudo-message row is a real Message row, so the existing
    # context pipeline (recent messages) picks it up with no changes.
    rows = (
        await session.execute(
            select(Message).where(Message.is_from_bot.is_(False))
        )
    ).scalars().all()
    assert any("[reacted 🥰 to your message]" == r.text for r in rows)


@pytest.mark.asyncio
async def test_reaction_row_enters_recent_context_window(
    session, session_factory_patch, allow_all_chats, bot
):
    conv = await get_or_create_conversation(session, -100999, "supergroup")
    await _seed_her_message(session, conv)
    await get_or_create_user(session, 555, "rae", "Rae", None)
    await session.commit()

    await bot.handle_reaction(
        _reaction_update(555, message_id=100, emoji="🥰"), None
    )

    # The pseudo-message row is a real Message row, so the existing
    # context pipeline (ConversationStateManager.get_context -> RECENT
    # MESSAGES in the prompt) picks it up with zero changes.
    from app.state import ConversationStateManager

    ctx = await ConversationStateManager(session).get_context(conv.id)
    texts = [m.text for m in ctx.recent_messages]
    assert "[reacted 🥰 to your message]" in texts
