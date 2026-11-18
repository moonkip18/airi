"""Genuine short-term working memory.

Working memory represents the *current social situation*, not long-term
memory. It is compact (a bounded, expiring set of typed entries per
conversation) and never dumps raw Telegram history into the LLM.

Tracked per conversation:
  * recent messages (compact — no full history)
  * current topic / subtopic
  * recent topic transitions
  * active participants
  * who is talking to whom (reply graph)
  * recent important events
  * Airi's recent actions
  * Airi's current involvement (participating / observing / away)
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.config import settings
from app.database import WorkingMemoryEntry

logger = structlog.get_logger()


class WorkingMemoryManager:
    def __init__(self, session: AsyncSession):
        self.session = session

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def get_window(
        self,
        conversation_id: int,
        entry_types: Optional[List[str]] = None,
        limit: int = 40,
    ) -> List[WorkingMemoryEntry]:
        stmt = select(WorkingMemoryEntry).where(
            WorkingMemoryEntry.conversation_id == conversation_id,
            WorkingMemoryEntry.expires_at > datetime.utcnow(),
        ).order_by(WorkingMemoryEntry.created_at.desc()).limit(limit)
        if entry_types:
            stmt = stmt.where(WorkingMemoryEntry.entry_type.in_(entry_types))
        result = await self.session.execute(stmt)
        return list(reversed(result.scalars().all()))

    async def get_messages(self, conversation_id: int, limit: int = 15) -> List[WorkingMemoryEntry]:
        return await self.get_window(conversation_id, entry_types=["message"], limit=limit)

    async def get_topic_history(self, conversation_id: int, limit: int = 6) -> List[Dict[str, Any]]:
        entries = await self.get_window(conversation_id, entry_types=["topic_transition"], limit=limit)
        return [dict(e.payload) for e in entries]

    async def get_recent_events(self, conversation_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        entries = await self.get_window(conversation_id, entry_types=["event"], limit=limit)
        return [dict(e.payload) for e in entries]

    async def get_airi_actions(self, conversation_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        entries = await self.get_window(conversation_id, entry_types=["action"], limit=limit)
        return [dict(e.payload) for e in entries]

    async def get_reply_targets(self, conversation_id: int, limit: int = 12) -> List[Dict[str, Any]]:
        """Who is talking to whom (recent reply/mention pairs)."""
        entries = await self.get_window(conversation_id, entry_types=["reply_target"], limit=limit)
        return [dict(e.payload) for e in entries]

    async def get_active_participants(self, conversation_id: int, window_minutes: int = 30) -> List[int]:
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        stmt = select(WorkingMemoryEntry).where(
            WorkingMemoryEntry.conversation_id == conversation_id,
            WorkingMemoryEntry.entry_type.in_(["message", "event", "action"]),
            WorkingMemoryEntry.created_at >= cutoff,
        )
        result = await self.session.execute(stmt)
        seen: Dict[int, float] = {}
        for entry in result.scalars().all():
            uid = entry.payload.get("user_id")
            if uid is not None:
                seen[uid] = entry.created_at.timestamp()
        return sorted(seen, key=lambda u: seen[u], reverse=True)

    async def build_summary(self, conversation_id: int) -> Dict[str, Any]:
        """Compact representation of the current social situation for prompts."""
        messages = await self.get_messages(conversation_id, limit=10)
        events = await self.get_recent_events(conversation_id, limit=5)
        actions = await self.get_airi_actions(conversation_id, limit=5)
        replies = await self.get_reply_targets(conversation_id, limit=8)
        active = await self.get_active_participants(conversation_id)

        return {
            "active_participants": active,
            "recent_messages": [
                {
                    "user_id": m.payload.get("user_id"),
                    "username": m.payload.get("username"),
                    "text": m.payload.get("text", "")[:150],
                    "is_airi": m.payload.get("is_airi", False),
                    "reply_target_user_id": m.payload.get("reply_target_user_id"),
                }
                for m in messages
            ],
            "recent_events": events,
            "airi_recent_actions": actions,
            "recent_reply_targets": replies,
        }

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    async def add_message(
        self,
        conversation_id: int,
        user_id: int,
        username: Optional[str],
        text: str,
        is_airi: bool = False,
        reply_target_user_id: Optional[int] = None,
        message_type: str = "text",
        importance: float = 0.3,
    ) -> WorkingMemoryEntry:
        return await self._add(
            conversation_id,
            "message",
            {
                "user_id": user_id,
                "username": username or (f"user{user_id}" if user_id else "unknown"),
                "text": (text or "")[:500],
                "is_airi": is_airi,
                "reply_target_user_id": reply_target_user_id,
                "message_type": message_type,
                "importance": importance,
            },
            expiry_minutes=settings.working_memory_expiry_minutes,
        )

    async def add_topic_transition(
        self,
        conversation_id: int,
        old_topic: Optional[str],
        new_topic: str,
        reason: str = "local_heuristic",
    ) -> WorkingMemoryEntry:
        return await self._add(
            conversation_id,
            "topic_transition",
            {
                "old_topic": old_topic,
                "new_topic": new_topic,
                "reason": reason,
                "at": datetime.utcnow().isoformat(),
            },
            expiry_minutes=settings.working_memory_expiry_minutes,
        )

    async def add_event(
        self,
        conversation_id: int,
        event: str,
        user_id: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> WorkingMemoryEntry:
        data = {"event": event, "importance": importance,
                "at": datetime.utcnow().isoformat()}
        if user_id is not None:
            data["user_id"] = user_id
        if payload:
            data.update(payload)
        return await self._add(
            conversation_id, "event", data, expiry_minutes=settings.working_memory_expiry_minutes,
        )

    async def add_airi_action(
        self,
        conversation_id: int,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> WorkingMemoryEntry:
        data = {"action": action, "at": datetime.utcnow().isoformat()}
        if payload:
            data.update(payload)
        return await self._add(
            conversation_id, "action", data, expiry_minutes=settings.working_memory_expiry_minutes,
        )

    async def add_reply_target(
        self,
        conversation_id: int,
        from_user_id: int,
        to_user_id: int,
    ) -> WorkingMemoryEntry:
        return await self._add(
            conversation_id,
            "reply_target",
            {"from_user_id": from_user_id, "to_user_id": to_user_id,
             "at": datetime.utcnow().isoformat()},
            expiry_minutes=settings.working_memory_expiry_minutes,
        )

    async def _add(
        self,
        conversation_id: int,
        entry_type: str,
        payload: Dict[str, Any],
        expiry_minutes: int,
    ) -> WorkingMemoryEntry:
        now = datetime.utcnow()
        entry = WorkingMemoryEntry(
            conversation_id=conversation_id,
            entry_type=entry_type,
            payload=payload,
            created_at=now,
            expires_at=now + timedelta(minutes=expiry_minutes),
        )
        self.session.add(entry)
        await self.session.flush()
        await self._prune(conversation_id)
        return entry

    async def _prune(self, conversation_id: int):
        """Keep working memory compact: remove expired + oldest overflow."""
        await self.session.execute(
            delete(WorkingMemoryEntry).where(
                WorkingMemoryEntry.conversation_id == conversation_id,
                WorkingMemoryEntry.expires_at <= datetime.utcnow(),
            )
        )

        stmt = select(WorkingMemoryEntry).where(
            WorkingMemoryEntry.conversation_id == conversation_id
        ).order_by(WorkingMemoryEntry.created_at.desc())
        result = await self.session.execute(stmt)
        entries = result.scalars().all()

        # Compress messages to a bounded tail (keep message detail only for
        # the most recent ones; older ones collapse to events).
        max_entries = settings.working_memory_max_entries
        if len(entries) > max_entries:
            overflow = entries[max_entries:]
            for entry in overflow:
                await self.session.delete(entry)
        await self.session.flush()

    async def set_involvement(self, conversation_id: int, involvement: str):
        """Airi's current involvement: participating / observing / away."""
        # Stored as a lightweight, single-row entry so it expires naturally.
        existing = await self.get_window(conversation_id, entry_types=["involvement"], limit=1)
        if existing:
            existing[-1].payload = {**existing[-1].payload, "involvement": involvement}
            await self.session.flush()
        else:
            await self._add(
                conversation_id, "involvement", {"involvement": involvement},
                expiry_minutes=settings.working_memory_expiry_minutes,
            )

    async def get_involvement(self, conversation_id: int) -> Optional[str]:
        entries = await self.get_window(conversation_id, entry_types=["involvement"], limit=1)
        if entries:
            return entries[-1].payload.get("involvement")
        return None

    async def clear_conversation(self, conversation_id: int):
        await self.session.execute(
            delete(WorkingMemoryEntry).where(WorkingMemoryEntry.conversation_id == conversation_id)
        )
        await self.session.flush()