from app.database.session import db_write_lock
import asyncio
import random
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from telegram import Update, Message, User as TelegramUser, Chat
from telegram.ext import (
    Application,
    MessageHandler,
    MessageReactionHandler,
    filters,
    ContextTypes,
)
from telegram.request import HTTPXRequest
from telegram.error import TimedOut, NetworkError

from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.config import settings
from app.database import get_session
from app.memory.store import get_or_create_user, get_or_create_conversation
from app.agent import Agent
from app.personality import get_personality
from app.social import SocialDecisionEngine
from app.state import EmotionalStateManager, RelationshipManager
from app.analysis import analyze_message, classify_target
from app.telegram.keyboards import create_keyboards


logger = structlog.get_logger()

# Per-actor cooldown for the relationship nudge a reaction earns. Reactions
# are cheap currency — someone spam-reacting every message shouldn't be able
# to farm affection. The emotional impact has its own shared dampening
# (EmotionalStateManager._RECENT_EMOTION_EVENTS).
_REACTION_NUDGE_COOLDOWN_SECONDS = 90
_recent_reaction_nudges: Dict[int, datetime] = {}


class TelegramBot:
    def __init__(self, token: str):
        self.token = token
        self.application: Optional[Application] = None
        self.bot_username: Optional[str] = None
        self.bot_id: Optional[int] = None

        # One shared decision engine for the whole bot: the rate limiter and
        # burst counters live on it and must survive across messages (a fresh
        # engine per message would make MAX_MESSAGES_PER_MINUTE useless).
        self._social_engine = SocialDecisionEngine()

        self._agent_cache: Dict[int, Agent] = {}
        self._background_tasks: Dict[int, asyncio.Task] = {}
        self._conversation_locks: Dict[int, "asyncio.Lock"] = {}
        self._pending_replies: set = set()

        self._initialized = False
        self._started = False
        # Set at the top of stop(): while shutting down, message handlers
        # that were queued behind locks can resume and would otherwise fire
        # brand-new reply tasks mid-shutdown ("ghost" replies appearing as
        # the process dies). Any decision made after this flag is set is
        # dropped instead of sent.
        self._stopping = False

    async def initialize(self):
        """
        Initialize the Telegram application and verify the bot connection.

        Telegram can be unreliable/unreachable depending on the network,
        so connection failures are retried instead of immediately killing
        the entire application.
        """

        logger.info("telegram_initializing")

        # Wire the configured proxy (TELEGRAM_PROXY in .env) into BOTH
        # request pools — without it, the setting existed but was silently
        # ignored and every connection went direct (fails on filtered
        # networks where Telegram is blocked).
        proxy = settings.telegram_proxy or None
        if proxy:
            logger.info("telegram_proxy_enabled", proxy=proxy)

        request = HTTPXRequest(
            connection_pool_size=16,
            connect_timeout=30.0,
            read_timeout=60.0,
            write_timeout=30.0,
            pool_timeout=30.0,
            proxy=proxy,
        )

        get_updates_request = HTTPXRequest(
            connection_pool_size=16,
            connect_timeout=30.0,
            read_timeout=90.0,
            write_timeout=30.0,
            pool_timeout=30.0,
            proxy=proxy,
        )

        self.application = (
            Application.builder()
            .token(self.token)
            .request(request)
            .get_updates_request(get_updates_request)
            # Process updates concurrently — PTB's default is strictly
            # one-at-a-time, so a burst of messages would each wait for the
            # previous one's full pipeline (including LLM calls) and she'd
            # fall behind. Per-conversation locks keep same-chat ordering.
            .concurrent_updates(256)
            .build()
        )

        # Register handlers before starting.
        # Broadened beyond plain text: stickers/photos/GIFs/voice are a huge
        # part of group-chat life — she should at least SEE them (as short
        # descriptors) instead of acting like they never happened.
        media_filters = (
            filters.TEXT
            | filters.PHOTO
            | filters.VIDEO
            | filters.ANIMATION
            | filters.VOICE
            | filters.VIDEO_NOTE
            | filters.AUDIO
            | filters.Document.ALL
            | filters.Sticker.ALL
        ) & ~filters.COMMAND & ~filters.StatusUpdate.ALL

        self.application.add_handler(
            MessageHandler(
                media_filters,
                self.handle_message,
            )
        )

        self.application.add_handler(
            MessageHandler(
                filters.StatusUpdate.NEW_CHAT_MEMBERS,
                self.handle_new_member,
            )
        )

        self.application.add_handler(
            MessageHandler(
                filters.StatusUpdate.LEFT_CHAT_MEMBER,
                self.handle_left_member,
            )
        )

        # She SEES reactions to messages (hers and others') but never
        # replies to them — a reaction is acknowledgment, not a prompt.
        # group=0 alongside MessageHandler: PTB only runs the first matching
        # handler per group, and these update types never overlap anyway.
        self.application.add_handler(
            MessageReactionHandler(self.handle_reaction),
        )

        self.application.add_error_handler(self.handle_error)

        # Explicitly initialize the Application before making API calls.
        await self.application.initialize()

        # Telegram may be temporarily unreachable. Retry several times.
        last_error = None

        for attempt in range(1, 6):
            try:
                logger.info(
                    "telegram_get_me_attempt",
                    attempt=attempt,
                    max_attempts=5,
                )

                bot_info = await self.application.bot.get_me(
                    read_timeout=60,
                    write_timeout=30,
                    connect_timeout=30,
                    pool_timeout=30,
                )

                self.bot_username = bot_info.username
                self.bot_id = bot_info.id

                logger.info(
                    "bot_initialized",
                    username=self.bot_username,
                    bot_id=bot_info.id,
                )

                self._initialized = True
                return

            except (TimedOut, NetworkError) as e:
                last_error = e

                logger.warning(
                    "telegram_connection_failed",
                    attempt=attempt,
                    max_attempts=5,
                    error=str(e),
                    proxy_configured=bool(settings.telegram_proxy),
                )

                if attempt < 5:
                    # Exponential-ish backoff:
                    # 2s, 4s, 6s, 8s
                    await asyncio.sleep(attempt * 2)

        logger.error(
            "telegram_initialization_failed",
            error=str(last_error),
        )

        # Clean up the Application because initialization failed.
        try:
            await self.application.shutdown()
        except Exception as e:
            logger.warning(
                "telegram_cleanup_failed",
                error=str(e),
            )

        self.application = None
        raise last_error

    async def start(self):
        if not self.application:
            raise RuntimeError("Telegram application has not been initialized")

        if self._started:
            return

        await self.application.start()
        await self.application.updater.start_polling(
    drop_pending_updates=True,
    # Telegram's default update set EXCLUDES message_reaction — without
    # this opt-in, reaction updates silently never arrive and she'd never
    # know anyone acknowledged her messages.
    allowed_updates=["message", "message_reaction"],
)
        self._started = True

        logger.info("bot_started")

    async def stop(self):
        """
        Safely stop the bot.

        This method is intentionally defensive because initialize()
        can fail before updater.start_polling() ever runs.
        """

        # Flag FIRST: queued message handlers resume as the background
        # tasks unwind (their cancellation also releases any lock they
        # held); without this they would spawn fresh reply tasks that
        # send during the shutdown window — the "replies suddenly appear
        # on Ctrl+C" bug.
        self._stopping = True

        # Cancel autonomous loops.
        for conversation_id, task in list(self._background_tasks.items()):
            task.cancel()

        self._background_tasks.clear()

        # Cancel pending delayed replies.
        for task in list(self._pending_replies):
            task.cancel()

        self._pending_replies.clear()

        if not self.application:
            return

        # Only stop the updater if we actually started it.
        if self._started:
            try:
                updater = self.application.updater

                if updater and updater.running:
                    await updater.stop()

            except Exception as e:
                logger.warning(
                    "updater_stop_failed",
                    error=str(e),
                )

        try:
            await self.application.stop()
        except RuntimeError:
            # Application may never have fully started.
            pass
        except Exception as e:
            logger.warning(
                "application_stop_failed",
                error=str(e),
            )

        try:
            await self.application.shutdown()
        except Exception as e:
            logger.warning(
                "application_shutdown_failed",
                error=str(e),
            )

        self._started = False
        self._initialized = False
        self.application = None

        logger.info("bot_stopped")

    def _get_agent(self, session: AsyncSession) -> Agent:
        return Agent(
            session,
            self.bot_username or "bot",
            social_engine=self._social_engine,
        )

    def _conversation_lock(self, conversation_id: int) -> "asyncio.Lock":
        lock = self._conversation_locks.get(conversation_id)

        if lock is None:
            lock = asyncio.Lock()
            self._conversation_locks[conversation_id] = lock

        return lock

    _MEDIA_LABELS = (
        (lambda m: m.sticker, "[sticker]"),
        (lambda m: m.photo, "[photo]"),
        (lambda m: m.animation, "[GIF]"),
        (lambda m: m.video_note, "[video message]"),
        (lambda m: m.video, "[video]"),
        (lambda m: m.voice, "[voice message]"),
        (lambda m: m.audio, "[audio]"),
        (lambda m: m.document, "[file]"),
    )

    def _media_descriptor(self, message: Message) -> Optional[str]:
        """Short descriptor for a non-text message, e.g. '[sticker]' or
        '[photo] nice view' (caption included). Returns None for message
        types she has no interest in (locations, contacts, service msgs)."""
        for getter, label in self._MEDIA_LABELS:
            try:
                if getter(message):
                    caption = (message.caption or "").strip()
                    return f"{label} {caption}".strip()
            except Exception:
                continue
        return None

    async def handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        message: Message = update.effective_message

        if not message:
            return

        if message.from_user.is_bot:
            return

        # Text, caption, or a media descriptor — she never sees raw bytes,
        # only a short textual stand-in for media.
        message_text = message.text or self._media_descriptor(message)
        if not message_text:
            return

        chat: Chat = update.effective_chat
        user: TelegramUser = message.from_user

        # Access control: when ALLOWED_CHAT_IDS is set, any other chat is
        # ignored BEFORE any analysis, DB write or LLM call — zero cost.
        # The chat id is logged so it can be copied into .env.
        from app.config import get_allowed_chat_ids
        allowed = get_allowed_chat_ids()
        if allowed is not None and chat.id not in allowed:
            logger.info(
                "chat_ignored_not_allowed",
                chat_id=chat.id,
                title=chat.title or "private",
                chat_type=chat.type,
            )
            return

        # NOTE: no LLM or network call ever runs under db_write_lock. But ALL
        # SQLite writes (including this ingestion block) are serialized
        # through it: SQLite's busy_timeout does NOT cover the WAL
        # read->write upgrade (instant SQLITE_BUSY_SNAPSHOT), so concurrent
        # handler transactions randomly collided into "database is locked"
        # cascades. One asyncio writer at a time makes contention impossible;
        # the section below is all cheap local work.
        async with get_session() as session:
            # db_write_lock scopes ONLY the ingestion + read phase: the
            # per-conversation lock and (critically) the LLM calls below
            # must never hold the global write lock — one slow generation
            # otherwise freezes message ingestion for EVERY chat (a 96s
            # LLM call produced a ~100s no-ingestion gap).
            await db_write_lock.acquire()
            try:
                db_user = await get_or_create_user(
                    session,
                    user.id,
                    user.username,
                    user.first_name,
                    user.last_name,
                )

                db_conversation = await get_or_create_conversation(
                    session,
                    chat.id,
                    chat.type,
                    chat.title,
                )

                is_direct_mention = self._is_direct_mention(message)

                reply_to_db_user_id = None
                reply_tg_message_id = None
                reply_context = None

                if message.reply_to_message and message.reply_to_message.from_user:
                    rt = message.reply_to_message.from_user

                    rt_user = await get_or_create_user(
                        session,
                        rt.id,
                        rt.username,
                        rt.first_name,
                        rt.last_name,
                    )

                    reply_to_db_user_id = rt_user.id
                    reply_tg_message_id = message.reply_to_message.message_id

                mentioned_db_ids = await self._resolve_mentioned_ids(
                    session,
                    message,
                )

                db_message_id = await self._store_message(
                    session,
                    db_conversation.id,
                    db_user.id,
                    message,
                    text=message_text,
                    reply_to_user_id=reply_to_db_user_id,
                )

                # Commit ingestion writes NOW, before the per-conversation lock:
                # that lock can wait 20-30s behind another message's LLM call,
                # and this session's uncommitted INSERT (an open SQLite write
                # transaction) held across the wait starves every other writer —
                # the root cause of the "database is locked" cascades.
                from app.database import commit_with_retry
                await commit_with_retry(session)

                # Resolve the actual message being replied to.
                if reply_tg_message_id is not None:
                    from app.database import Message as DBMessage
                    from sqlalchemy import select as _select

                    quoted = (
                        await session.execute(
                            _select(DBMessage)
                            .where(
                                DBMessage.conversation_id == db_conversation.id,
                                DBMessage.telegram_message_id == reply_tg_message_id,
                            )
                            .order_by(DBMessage.id.desc())
                            .limit(1)
                        )
                    ).scalar_one_or_none()

                    if quoted is not None:
                        reply_context = {
                            "user_id": quoted.user_id,
                            "is_from_bot": bool(quoted.is_from_bot),
                            "text": (quoted.text or "")[:300],
                        }
                    else:
                        reply_context = {
                            "user_id": reply_to_db_user_id,
                            "text": None,
                        }

                bot_db_user_id = await self._get_bot_user_id(session)

                analysis = analyze_message(
                    message_text,
                    message_id=message.message_id,
                    speaker_user_id=db_user.id,
                    speaker_username=user.username,
                    reply_target_message_id=reply_tg_message_id,
                    reply_target_user_id=reply_to_db_user_id,
                    mentioned_user_ids=mentioned_db_ids,
                )

                target = classify_target(
                    is_direct_mention=is_direct_mention,
                    reply_target_user_id=reply_to_db_user_id,
                    speaker_user_id=db_user.id,
                    bot_user_id=bot_db_user_id,
                    mentioned_user_ids=mentioned_db_ids,
                    message_type=analysis.message_type,
                    text=message_text,
                )

                agent = self._get_agent(session)
            finally:
                db_write_lock.release()


            # Serialize per conversation.
            lock = self._conversation_lock(
                db_conversation.id
            )

            async with lock:
                decision = await agent.process_message(
                    conversation_id=db_conversation.id,
                    user=db_user,
                    message_text=message_text,
                    is_direct_mention=is_direct_mention,
                    reply_to_user_id=reply_to_db_user_id,
                    telegram_message_id=message.message_id,
                    reply_to_message_id=reply_tg_message_id,
                    mentioned_user_ids=mentioned_db_ids,
                    message_analysis=analysis,
                    target=target,
                    bot_user_id=bot_db_user_id,
                    reply_context=reply_context,
                    db_message_id=db_message_id,
                )

                if decision:
                    await self._handle_decision(
                        chat.id,
                        decision,
                        message.message_id,
                        db_conversation.id,
                        force=is_direct_mention,
                    )

                # Release the write transaction BEFORE the lock can be held
                # by the next waiter / before any slow work below: a session
                # that errored mid-decision must not leave an open tx behind.
                await commit_with_retry(session)

    async def _handle_decision(
        self,
        chat_id: int,
        decision,
        message_id: int,
        db_conversation_id: int,
        force: bool = False,
        in_response_to_db_id: Optional[int] = None,
    ):
        if self._stopping:
            # Shutdown in progress: never spawn new sends. See _stopping.
            logger.info(
                "reply_dropped_shutdown",
                chat_id=chat_id,
                message_id=message_id,
            )
            return

        action = (
            decision.action.value
            if hasattr(decision.action, "value")
            else decision.action
        )

        if action in ("reply", "initiate", "leave") and decision.response:
            task = asyncio.create_task(
                self._send_delayed_response(
                    chat_id,
                    decision.response,
                    decision.response_delay,
                    reply_to_message_id=message_id if action == "reply" else None,
                    db_conversation_id=db_conversation_id,
                    force=force or action == "leave",
                    in_response_to_db_id=in_response_to_db_id,
                    double_text=getattr(decision, "double_text", "") or "",
                )
            )

            self._pending_replies.add(task)
            task.add_done_callback(self._pending_replies.discard)

        elif action == "react" and decision.reaction_emoji:
            task = asyncio.create_task(
                self._send_reaction(
                    chat_id,
                    message_id,
                    decision.reaction_emoji,
                )
            )

            self._pending_replies.add(task)
            task.add_done_callback(self._pending_replies.discard)

    async def _send_reaction(
        self,
        chat_id: int,
        message_id: int,
        emoji: str,
    ):
        await asyncio.sleep(0.3)

        try:
            # PTB requires proper ReactionType objects — raw dicts blow up
            # inside PTB internals ("cannot use 'dict' as a set element").
            from telegram import ReactionTypeEmoji

            await self.application.bot.set_message_reaction(
                chat_id=chat_id,
                message_id=message_id,
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )

            logger.info(
                "reaction_sent",
                chat_id=chat_id,
                message_id=message_id,
                emoji=emoji,
            )

        except Exception as e:
            logger.error(
                "reaction_send_failed",
                error=str(e),
                chat_id=chat_id,
            )

    async def _resolve_mentioned_ids(
        self,
        session: AsyncSession,
        message: Message,
    ) -> list:
        """Resolve Telegram mentions to internal DB user IDs."""

        from app.database import User as DBUser
        from sqlalchemy import select

        mentioned = []
        text = message.text or ""

        for ent in (message.entities or []):
            if (
                getattr(ent, "type", None) == "text_mention"
                and getattr(ent, "user", None)
            ):
                u = ent.user

                du = await get_or_create_user(
                    session,
                    u.id,
                    u.username,
                    u.first_name,
                    u.last_name,
                )

                mentioned.append(du.id)

            elif getattr(ent, "type", None) == "mention":
                raw = text[
                    ent.offset:
                    ent.offset + ent.length
                ].lstrip("@")

                if raw:
                    stmt = select(DBUser).where(
                        DBUser.username == raw
                    )

                    result = await session.execute(stmt)

                    for du in result.scalars().all():
                        mentioned.append(du.id)

        return mentioned

    def _is_direct_mention(self, message: Message) -> bool:
        if not self.bot_username:
            return False

        text = (message.text or "").strip()

        if not text:
            return False

        # @Airi / @airi (the actual bot handle)
        if f"@{self.bot_username}".lower() in text.lower():
            return True

        # Plain name anywhere in the message ("airi you're so cute").
        # The name comes from the personality config so it always matches
        # the persona she's actually playing.
        try:
            persona_name = (get_personality().character.get("name") or self.bot_username)
        except Exception:
            persona_name = self.bot_username

        persona_name = str(persona_name).strip().lower()
        if persona_name and re.search(
            rf"(?<!\w){re.escape(persona_name)}(?!\w)",
            text,
            re.IGNORECASE,
        ):
            return True

        # Replying directly to her.
        if (
            message.reply_to_message
            and message.reply_to_message.from_user
        ):
            username = message.reply_to_message.from_user.username

            if username:
                if username.lower() == self.bot_username.lower():
                    return True

        return False

    async def _store_message(
        self,
        session: AsyncSession,
        conversation_id: int,
        user_id: int,
        message: Message,
        text: Optional[str] = None,
        reply_to_user_id: Optional[int] = None,
    ) -> int:
        from app.database import Message as DBMessage

        db_message = DBMessage(
            conversation_id=conversation_id,
            user_id=user_id,
            telegram_message_id=message.message_id,
            text=text if text is not None else message.text,
            reply_to_message_id=(
                message.reply_to_message.message_id
                if message.reply_to_message
                else None
            ),
            reply_to_user_id=reply_to_user_id,
            timestamp=datetime.utcnow(),
            is_from_bot=False,
        )

        session.add(db_message)
        # Flush now so the caller gets the internal id — the stale-reply
        # check needs to know exactly which message she is replying to.
        await session.flush()
        return db_message.id

    async def _store_bot_message(
        self,
        session: AsyncSession,
        conversation_id: int,
        text: str,
        telegram_message_id: Optional[int] = None,
    ):
        """Persist Airi's own replies.

        Storing the REAL Telegram message id (not a placeholder) is what lets
        future reply-to-bot messages resolve the exact message she sent —
        without it, she can't tell which of her messages someone answered.
        """

        from app.database import Message as DBMessage

        bot_user_id = await self._get_bot_user_id(session)

        db_message = DBMessage(
            conversation_id=conversation_id,
            user_id=bot_user_id,
            telegram_message_id=telegram_message_id or 0,
            text=text,
            reply_to_message_id=None,
            timestamp=datetime.utcnow(),
            is_from_bot=True,
        )

        session.add(db_message)

    async def _get_bot_user_id(
        self,
        session: AsyncSession,
    ) -> int:
        bot_username = self.bot_username or "airi"

        # Stable sentinel ID for Airi.
        sentinel_id = 777000000

        user = await get_or_create_user(
            session,
            sentinel_id,
            bot_username,
            "Airi",
            None,
        )

        return user.id

    # ----------------------------------------------------------------------
    # Human-like delivery: think delay -> staleness check -> typing -> send
    # (sometimes as several short bubbles, with a rare self-correction).
    # ----------------------------------------------------------------------

    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")

    @staticmethod
    def _split_response(text: str) -> list:
        """Split a response into natural chat bubbles.

        Measured in her own logs: humans split one thought into 2+ rapid
        messages ~16% of the time; she did it ~8%. So the bar for splitting
        is low: any multi-sentence reply over ~50 chars gets fired off as
        separate bubbles — "wait what. ok explain" reads as two messages,
        not one composed line. Max 3 bubbles; leftovers merge into the last.
        """
        text = (text or "").strip()
        if not text:
            return []

        if len(text) <= 50 or "\n" not in text and len(text) <= 50:
            return [text]

        sentences = [s.strip() for s in TelegramBot._SENTENCE_SPLIT.split(text) if s.strip()]
        if len(sentences) < 2:
            return [text]

        bubbles: list = []
        buf = ""
        for s in sentences:
            if buf and len(buf) + len(s) + 1 > 90:
                bubbles.append(buf)
                buf = s
            else:
                buf = f"{buf} {s}".strip()
            if len(bubbles) == 2 and buf:
                # Last allowed bubble absorbs the rest.
                break
        if buf:
            if len(bubbles) >= 3:
                bubbles[-1] = f"{bubbles[-1]} {buf}".strip()
            else:
                bubbles.append(buf)

        # A tiny trailing fragment looks broken; merge it back.
        if len(bubbles) >= 2 and len(bubbles[-1]) < 4:
            bubbles[-2] = f"{bubbles[-2]} {bubbles[-1]}".strip()
            bubbles.pop()

        return bubbles if len(bubbles) > 1 else [text]

    @staticmethod
    def _make_typo(text: str) -> Optional[str]:
        """Introduce ONE human-looking typo (swapped adjacent letters)."""
        words = re.findall(r"[A-Za-z]{4,}", text)
        candidates = [w for w in words if len(w) >= 4]
        if not candidates:
            return None
        word = random.choice(candidates)
        idx = random.randint(1, len(word) - 3)
        a, b = word[idx], word[idx + 1]
        if a == b:
            return None
        swapped = word[:idx] + b + a + word[idx + 2:]
        # Replace only the first occurrence, preserving the rest of the text.
        return text.replace(word, swapped, 1)

    async def _reply_is_stale(
        self,
        db_conversation_id: Optional[int],
        in_response_to_db_id: Optional[int],
    ) -> bool:
        """Has the conversation moved on while she was 'thinking'?

        Humans abandon replies that are suddenly irrelevant. She drops a
        response when: her own message is the latest thing in the chat
        (she'd be talking to myself), or new messages arrived after the one
        she was answering (the conversation left her behind — answering the
        old moment now reads as a bot, not a person). Direct mentions are
        never dropped — people wait for those.
        """
        if db_conversation_id is None:
            return False

        try:
            async with get_session() as session:
                from app.database import Message as DBMessage
                from sqlalchemy import select as _select

                stmt = (
                    _select(DBMessage)
                    .where(DBMessage.conversation_id == db_conversation_id)
                    .order_by(DBMessage.id.desc())
                    .limit(5)
                )
                recent = (await session.execute(stmt)).scalars().all()

            if not recent:
                return False

            newest = recent[0]
            if newest.is_from_bot:
                # She already has the last word — sending more reads as nagging.
                return True

            if in_response_to_db_id is None:
                return False

            newer = sum(
                1 for m in recent
                if m.id > in_response_to_db_id and not m.is_from_bot
            )
            return newer >= 2

        except Exception as e:
            # If the check itself fails, prefer to speak rather than go mute.
            logger.warning("stale_reply_check_failed", error=str(e))
            return False

    async def _send_delayed_response(
        self,
        chat_id: int,
        text: str,
        delay: float,
        reply_to_message_id: Optional[int] = None,
        db_conversation_id: Optional[int] = None,
        force: bool = False,
        in_response_to_db_id: Optional[int] = None,
        double_text: str = "",
    ):
        try:
            # --- "Thinking" pause + typing indicator ----------------------
            # A human starts TYPING before the message lands, so split the
            # delay: quiet thinking first, then the typing indicator for the
            # last stretch. (Telegram's indicator expires after ~5s, so
            # triggering it at the start of a long delay would expire unused.)
            typing_time = random.uniform(0.9, 2.2)
            if delay < 1.2:
                # Fast replies (direct mentions): keep them snappy.
                typing_time = min(0.7, delay * 0.6)

            if delay > typing_time:
                await asyncio.sleep(delay - typing_time)
            # (if delay <= typing_time, the whole delay was "thinking")

            # --- Conversation moved on? Then a human would just not send it.
            if not force and await self._reply_is_stale(
                db_conversation_id, in_response_to_db_id,
            ):
                logger.info(
                    "reply_dropped_stale",
                    chat_id=chat_id,
                    text_preview=text[:60],
                )
                return

            # Show typing before the FIRST bubble.
            try:
                await self.application.bot.send_chat_action(
                    chat_id=chat_id, action="typing",
                )
            except Exception:
                pass

            if typing_time > 0:
                await asyncio.sleep(typing_time)

            # Typo behavior is state-driven: energy raises the base rate
            # (fast sloppy typing), boredom lowers it (deliberate, checked).
            typo_chance = settings.typo_chance_base
            if db_conversation_id is not None:
                try:
                    async with get_session() as s:
                        emo = await EmotionalStateManager(s).get_state()
                    if emo.get("energy", 0.5) >= 0.65:
                        typo_chance *= 2.0
                    elif emo.get("energy", 0.5) <= 0.3:
                        typo_chance *= 0.5
                except Exception:
                    pass

            # --- Deliver, possibly as several quick bubbles ---------------
            # The profanity filter runs here at send time — the LLM writes
            # naturally, the platform bleeps the hard words.
            from app.social.censor import censor
            bubbles = self._split_response(censor(text))
            first_message_id = None

            for i, bubble in enumerate(bubbles):
                if i > 0:
                    # Small gap + typing indicator between bubbles, like
                    # someone firing off a second thought.
                    try:
                        await self.application.bot.send_chat_action(
                            chat_id=chat_id, action="typing",
                        )
                    except Exception:
                        pass
                    await asyncio.sleep(random.uniform(0.7, 1.8))

                outgoing = bubble
                correction = None
                # Human touch: an occasional typo. Critically, the typo is
                # NOT always fixed — people leave typos standing, notice
                # them late, or never. Correction is a separate dice roll
                # (and late corrections land after a human-sized gap).
                if (
                    len(bubble) >= 20
                    and random.random() < typo_chance
                ):
                    typo = self._make_typo(bubble)
                    if typo:
                        outgoing = typo
                        correction = bubble

                sent = await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=outgoing,
                    reply_to_message_id=reply_to_message_id if i == 0 else None,
                    disable_web_page_preview=True,
                )
                if i == 0:
                    first_message_id = getattr(sent, "message_id", None)

                logger.info(
                    "response_sent",
                    chat_id=chat_id,
                    text_preview=bubble[:100],
                    delay=round(delay, 1) if i == 0 else None,
                    bubble=f"{i + 1}/{len(bubbles)}",
                )

                if correction and getattr(sent, "message_id", None):
                    if random.random() < settings.typo_correction_chance:
                        # She notices — sometimes quickly, sometimes a
                        # conversation-beat later, like a real person.
                        await asyncio.sleep(random.uniform(1.5, 6.0))
                        try:
                            await self.application.bot.edit_message_text(
                                chat_id=chat_id,
                                message_id=sent.message_id,
                                text=correction,
                            )
                            logger.info(
                                "self_corrected_typo",
                                chat_id=chat_id,
                                message_id=sent.message_id,
                            )
                        except Exception:
                            correction = outgoing  # store what actually stands
                    else:
                        # She did NOT notice right away (#5): like a real
                        # person, she may spot it much later and silently
                        # edit — no comment, no asterisk, just fixed.
                        if random.random() < settings.late_typo_fix_chance:
                            asyncio.create_task(self._late_typo_fix(
                                chat_id,
                                sent.message_id,
                                correction,
                                db_conversation_id,
                            ))
                        else:
                            logger.info(
                                "typo_left_standing",
                                chat_id=chat_id,
                                message_id=sent.message_id,
                            )
                        correction = None

                # Persist what she actually said (final wording per bubble).
                if db_conversation_id is not None:
                    async with db_write_lock, get_session() as session:
                        await self._store_bot_message(
                            session,
                            db_conversation_id,
                            correction or outgoing,
                            telegram_message_id=getattr(sent, "message_id", None),
                        )
                        await session.commit()

            # --- Double-text ----------------------------------------------
            # She sometimes fires a small follow-up ~30s later — but ONLY
            # if the chat stayed quiet after her message (that's what makes
            # it a human double-text instead of an interruption).
            if (
                double_text.strip()
                and db_conversation_id is not None
                and first_message_id
                and random.random() < 0.5
            ):
                wait_s = random.uniform(
                    settings.double_text_min_seconds,
                    settings.double_text_max_seconds,
                )
                dt_task = asyncio.create_task(self._maybe_send_double_text(
                    chat_id,
                    double_text.strip()[:120],
                    db_conversation_id,
                    first_message_id,
                    wait_s,
                ))
                self._pending_replies.add(dt_task)
                dt_task.add_done_callback(self._pending_replies.discard)

        except asyncio.CancelledError:
            raise

        except Exception as e:
            logger.error(
                "send_response_failed",
                error=str(e),
                chat_id=chat_id,
            )

    async def _late_typo_fix(
        self,
        chat_id: int,
        message_id: int,
        corrected_text: str,
        db_conversation_id: Optional[int],
    ):
        """#5: she spots her own typo a while later and silently edits it.

        The delayed notice is the point — an instant fix reads as a
        proofreader, a fix 30-90s later reads as a person who glanced back
        at their own message. DB row is updated too, so reply-chain context
        matches what's actually on screen.
        """
        try:
            wait_s = random.uniform(
                settings.late_typo_fix_min_seconds,
                settings.late_typo_fix_max_seconds,
            )
            await asyncio.sleep(wait_s)
            if self._stopping:
                return
            await self.application.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=corrected_text,
            )
            logger.info(
                "late_typo_fix",
                chat_id=chat_id,
                message_id=message_id,
                delay=round(wait_s, 1),
            )
            if db_conversation_id is not None:
                async with db_write_lock, get_session() as session:
                    from app.database import Message as DBMessage
                    from sqlalchemy import select as _select
                    row = (await session.execute(
                        _select(DBMessage).where(
                            DBMessage.conversation_id == db_conversation_id,
                            DBMessage.telegram_message_id == message_id,
                            DBMessage.is_from_bot.is_(True),
                        ).order_by(DBMessage.id.desc()).limit(1)
                    )).scalar_one_or_none()
                    if row is not None:
                        row.text = corrected_text
                    await session.commit()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # A failed edit is invisible to the chat — just log it.
            logger.debug("late_typo_fix_failed", error=str(e), chat_id=chat_id)

    async def _maybe_send_double_text(
        self,
        chat_id: int,
        text: str,
        db_conversation_id: int,
        after_message_id: int,
        wait_s: float,
    ):
        """Deliver the double-text only if nobody spoke after her message.

        Someone replying (or any new traffic) kills it — a person who
        double-texts into an active chat looks desperate; into a silent
        chat it's just... a person.
        """
        try:
            await asyncio.sleep(wait_s)
            if self._stopping:
                return

            async with get_session() as session:
                from app.database import Message as DBMessage
                from sqlalchemy import select as _select, func as _func

                newer = await session.scalar(
                    _select(_func.count(DBMessage.id)).where(
                        DBMessage.conversation_id == db_conversation_id,
                        DBMessage.telegram_message_id > after_message_id,
                        DBMessage.is_from_bot.is_(False),
                    )
                ) or 0

            if newer > 0:
                logger.info(
                    "double_text_cancelled_chat_active",
                    chat_id=chat_id,
                )
                return

            try:
                await self.application.bot.send_chat_action(
                    chat_id=chat_id, action="typing",
                )
            except Exception:
                pass
            await asyncio.sleep(random.uniform(0.8, 1.6))

            from app.social.censor import censor
            await self.application.bot.send_message(
                chat_id=chat_id, text=censor(text), disable_web_page_preview=True,
            )
            logger.info(
                "double_text_sent",
                chat_id=chat_id,
                text_preview=text[:60],
            )

            if db_conversation_id is not None:
                async with db_write_lock, get_session() as session:
                    await self._store_bot_message(
                        session,
                        db_conversation_id,
                        text,
                    )
                    await session.commit()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                "double_text_failed",
                error=str(e),
                chat_id=chat_id,
            )

    # Low-key welcomes — her own register, never a speech. A kid saying hi,
    # not a concierge.
    WELCOMES = ["yo", "welcome", "oh hey", "hi", "ay new person"]

    async def handle_new_member(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        if update.message is None:
            return

        chat: Chat = update.effective_chat

        # Access control: ignore joins in chats she doesn't work in.
        from app.config import get_allowed_chat_ids
        allowed = get_allowed_chat_ids()
        if allowed is not None and chat.id not in allowed:
            return

        for member in update.message.new_chat_members:
            if member.username == self.bot_username:
                logger.info(
                    "bot_added_to_chat",
                    chat_id=chat.id,
                )

                async with db_write_lock, get_session() as session:
                    db_conversation = await get_or_create_conversation(
                        session,
                        chat.id,
                        chat.type,
                        chat.title,
                    )

                    from app.database import ConversationState
                    from app.state import ConversationStateManager

                    manager = ConversationStateManager(session)

                    await manager.update_state(
                        db_conversation.id,
                        new_state=ConversationState.OBSERVING,
                        is_observing=True,
                    )

                    await session.commit()
                continue

            if member.is_bot:
                continue

            # She SEES the join: it enters recent messages as a real event
            # ("[joined the group]"), so it shapes context like anything else.
            try:
                async with db_write_lock, get_session() as session:
                    db_user = await get_or_create_user(
                        session,
                        member.id,
                        member.username,
                        member.first_name,
                        member.last_name,
                    )
                    db_conv = await get_or_create_conversation(
                        session,
                        chat.id,
                        chat.type,
                        chat.title,
                    )
                    from app.database import Message as DBMessage
                    session.add(DBMessage(
                        conversation_id=db_conv.id,
                        user_id=db_user.id,
                        telegram_message_id=update.message.message_id,
                        text="[joined the group]",
                        timestamp=datetime.utcnow(),
                        is_from_bot=False,
                    ))
                    await session.commit()

                # Occasional low-key welcome (local, no LLM). Mostly she just
                # notices — humans don't greet every joiner either.
                import random as _random
                if _random.random() > 0.45:
                    continue
                if self._social_engine.is_rate_limited():
                    continue

                name = member.first_name or member.username or "you"
                text = f"{_random.choice(self.WELCOMES)} {name}"

                sent = await self.application.bot.send_message(
                    chat_id=chat.id, text=text,
                )
                self._social_engine.record_message()

                async with db_write_lock, get_session() as session:
                    await self._store_bot_message(
                        session,
                        db_conv.id,
                        text,
                        telegram_message_id=getattr(sent, "message_id", None),
                    )
                    await session.commit()

                logger.info(
                    "member_welcomed",
                    chat_id=chat.id,
                    member=name,
                )

            except Exception as e:
                logger.warning(
                    "member_join_handling_failed",
                    error=str(e),
                )

    async def handle_left_member(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        if update.message is None or update.message.left_chat_member is None:
            return

        chat: Chat = update.effective_chat

        from app.config import get_allowed_chat_ids
        allowed = get_allowed_chat_ids()
        if allowed is not None and chat.id not in allowed:
            return

        left = update.message.left_chat_member

        if left.username == self.bot_username:
            logger.info(
                "bot_removed_from_chat",
                chat_id=chat.id,
            )
            return

        if left.is_bot:
            return

        # She notices the departure in context ("[left the group]") but
        # doesn't comment — people leave, that's normal.
        try:
            async with db_write_lock, get_session() as session:
                db_user = await get_or_create_user(
                    session,
                    left.id,
                    left.username,
                    left.first_name,
                    left.last_name,
                )
                db_conv = await get_or_create_conversation(
                    session,
                    chat.id,
                    chat.type,
                    chat.title,
                )
                from app.database import Message as DBMessage
                session.add(DBMessage(
                    conversation_id=db_conv.id,
                    user_id=db_user.id,
                    telegram_message_id=update.message.message_id,
                    text="[left the group]",
                    timestamp=datetime.utcnow(),
                    is_from_bot=False,
                ))
                await session.commit()
        except Exception as e:
            logger.warning(
                "member_left_handling_failed",
                error=str(e),
            )

    # Heart-family reactions read as stronger warmth than the rest.
    _LOVE_REACTIONS = {"❤", "😍", "🥰", "❤‍🔥"}

    async def handle_reaction(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        """Perceive reactions to messages.

        A reaction becomes a pseudo-message row ("[reacted ❤ to your
        message]") — the same pattern as "[joined the group]" — so it
        enters her context and, via the extraction loop, long-term memory.
        It never triggers the decision pipeline: a person who says
        "thanks for the like" reads as a bot.

        The critical part is `reply_to_user_id` on that row: when someone
        reacts to HER message, the row points back at her internal id, and
        `_check_being_ignored` (agent core) counts it as engagement. A
        heart in place of an answer is attention, not being ignored —
        exactly the failure this closes.
        """
        rxn = update.message_reaction
        if rxn is None:
            return

        # Anonymous channel reactions arrive via actor_chat with user=None;
        # those carry no person worth remembering — skip them.
        actor = rxn.user
        if actor is None:
            logger.info(
                "reaction_seen_anonymous",
                chat_id=rxn.chat.id if rxn.chat else None,
            )
            return

        # Her own reactions echo back as updates (she REACTs via the
        # decision engine) — they must never register as social input.
        if actor.is_bot or actor.id == self.bot_id:
            return

        chat: Chat = rxn.chat

        from app.config import get_allowed_chat_ids
        allowed = get_allowed_chat_ids()
        if allowed is not None and chat.id not in allowed:
            return

        new = tuple(rxn.new_reaction or ())
        emoji = next(
            (r.emoji for r in new if getattr(r, "emoji", None)),
            "[custom emoji]" if new else None,
        )

        if emoji is None:
            # Reaction taken back. She notices (log) but nothing is worth
            # remembering — no row, no nudge.
            logger.info(
                "reaction_removed",
                chat_id=chat.id,
                message_id=rxn.message_id,
            )
            return

        try:
            async with db_write_lock, get_session() as session:
                db_actor = await get_or_create_user(
                    session,
                    actor.id,
                    actor.username,
                    actor.first_name,
                    actor.last_name,
                )

                db_conversation = await get_or_create_conversation(
                    session,
                    chat.id,
                    chat.type,
                    chat.title,
                )

                from app.database import Message as DBMessage, User as DBUser
                from sqlalchemy import select as _select

                target = (
                    await session.execute(
                        _select(DBMessage)
                        .where(
                            DBMessage.conversation_id == db_conversation.id,
                            DBMessage.telegram_message_id == rxn.message_id,
                        )
                        .order_by(DBMessage.id.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()

                on_hers = bool(target and target.is_from_bot)

                if target is not None:
                    if target.user_id == db_actor.id:
                        # Reacting to your own message — she doesn't judge.
                        return
                    reply_to_user_id = target.user_id
                    if on_hers:
                        text = f"[reacted {emoji} to your message]"
                    else:
                        author = await session.get(DBUser, target.user_id)
                        name = (
                            (author.username if author and author.username else None)
                            or (author.first_name if author else None)
                            or "someone"
                        )
                        text = f"[reacted {emoji} to {name}'s message]"
                else:
                    # Reacted to a message older than her history — she saw
                    # it happen, she just can't attribute it.
                    reply_to_user_id = None
                    text = f"[reacted {emoji}]"

                session.add(DBMessage(
                    conversation_id=db_conversation.id,
                    user_id=db_actor.id,
                    telegram_message_id=rxn.message_id,
                    text=text,
                    reply_to_message_id=rxn.message_id,
                    reply_to_user_id=reply_to_user_id,
                    timestamp=datetime.utcnow(),
                    is_from_bot=False,
                ))

                if on_hers:
                    # Quiet warmth: being acknowledged feels good even when
                    # she has nothing to say back.
                    intensity = 1.0 if emoji in self._LOVE_REACTIONS else 0.6
                    await EmotionalStateManager(session).update_from_event(
                        "reaction_received", intensity,
                    )

                    # Relationship nudge, gated by a per-actor cooldown so
                    # spam-reacting everything can't farm affection.
                    now = datetime.utcnow()
                    last_nudge = _recent_reaction_nudges.get(actor.id)
                    if (
                        last_nudge is None
                        or (now - last_nudge).total_seconds()
                        >= _REACTION_NUDGE_COOLDOWN_SECONDS
                    ):
                        _recent_reaction_nudges[actor.id] = now
                        if len(_recent_reaction_nudges) > 200:
                            cutoff = now - timedelta(
                                seconds=_REACTION_NUDGE_COOLDOWN_SECONDS * 4
                            )
                            for key, ts in list(_recent_reaction_nudges.items()):
                                if ts < cutoff:
                                    del _recent_reaction_nudges[key]

                        await RelationshipManager(session).update_from_interaction(
                            db_actor.id, "positive", 0.5,
                        )

                await session.commit()

            logger.info(
                "reaction_seen",
                chat_id=chat.id,
                message_id=rxn.message_id,
                emoji=emoji,
                user_id=actor.id,
                on_hers=on_hers,
            )

        except Exception as e:
            logger.warning(
                "reaction_handling_failed",
                error=str(e),
                chat_id=chat.id,
            )

    async def handle_error(
        self,
        update: object,
        context: ContextTypes.DEFAULT_TYPE,
    ):
        logger.error(
            "telegram_handler_error",
            error=str(context.error),
            update_type=type(update).__name__,
        )

    async def start_autonomous_loop(
        self,
        conversation_id: int,
    ):
        if self._stopping or conversation_id in self._background_tasks:
            return

        async def loop():
            while True:
                try:
                    await asyncio.sleep(
                        settings.autonomous_check_interval
                    )

                    # No global db_write_lock: autonomous_check includes an
                    # LLM call and must not block message handling everywhere.
                    async with get_session() as session:
                        from app.database import Conversation
                        from sqlalchemy import select

                        stmt = select(Conversation).where(
                            Conversation.id == conversation_id
                        )

                        conv = (
                            await session.execute(stmt)
                        ).scalar_one_or_none()

                        if conv is None:
                            await self.stop_autonomous_loop(
                                conversation_id
                            )
                            break

                        agent = self._get_agent(session)

                        decision = await agent.autonomous_check(
                            conversation_id
                        )

                        if decision and decision.response:
                            # Commit the post-decision writes BEFORE the send:
                            # _send_delayed_response sleeps + does network I/O,
                            # and an uncommitted write tx held across that is
                            # what caused "database is locked" cascades.
                            await session.commit()

                            await self._send_delayed_response(
                                conv.telegram_chat_id,
                                decision.response,
                                decision.response_delay,
                                db_conversation_id=conversation_id,
                            )

                            await session.commit()

                except asyncio.CancelledError:
                    break

                except Exception as e:
                    logger.error(
                        "autonomous_loop_error",
                        error=str(e),
                        conversation_id=conversation_id,
                    )

        self._background_tasks[conversation_id] = (
            asyncio.create_task(loop())
        )

        logger.info(
            "autonomous_loop_started",
            conversation_id=conversation_id,
        )

    async def stop_autonomous_loop(
        self,
        conversation_id: int,
    ):
        task = self._background_tasks.get(conversation_id)

        if task:
            task.cancel()
            self._background_tasks.pop(
                conversation_id,
                None,
            )

            logger.info(
                "autonomous_loop_stopped",
                conversation_id=conversation_id,
            )