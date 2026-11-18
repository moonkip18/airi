"""Delayed action system.

A lightweight scheduler for future Airi actions. An action is represented as:

    action_type, execute_at, conversation/group, reason/context, state

Examples:
  * delayed_reaction  -> react to a message a bit later (rare)
  * return_to_conversation -> re-enter a conversation she left
  * follow_up         -> check back on an earlier topic
  * delayed_reply     -> a staged reply to something she noticed

Scheduling is non-blocking: waiting happens in the background loop, and new
incoming messages are processed normally.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.database import ScheduledAction

logger = structlog.get_logger()


class DelayedActionScheduler:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def schedule(
        self,
        conversation_id: Optional[int],
        action_type: str,
        execute_at: datetime,
        payload: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
    ) -> ScheduledAction:
        action = ScheduledAction(
            conversation_id=conversation_id,
            action_type=action_type,
            execute_at=execute_at,
            payload=payload or {},
            state="pending",
            reason=reason,
        )
        self.session.add(action)
        await self.session.flush()
        logger.info(
            "delayed_action_scheduled",
            action_type=action_type,
            conversation_id=conversation_id,
            execute_at=execute_at.isoformat(),
            reason=reason,
        )
        return action

    async def schedule_in(
        self,
        conversation_id: Optional[int],
        action_type: str,
        delay_seconds: float,
        payload: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
    ) -> ScheduledAction:
        return await self.schedule(
            conversation_id,
            action_type,
            datetime.utcnow() + timedelta(seconds=delay_seconds),
            payload,
            reason,
        )

    async def get_due(self, limit: int = 20) -> List[ScheduledAction]:
        stmt = select(ScheduledAction).where(
            ScheduledAction.state == "pending",
            ScheduledAction.execute_at <= datetime.utcnow(),
        ).order_by(ScheduledAction.execute_at.asc()).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def mark_executed(self, action: ScheduledAction):
        action.state = "executed"
        action.executed_at = datetime.utcnow()
        await self.session.flush()

    async def cancel(self, action_id: int):
        await self.session.execute(
            update(ScheduledAction)
            .where(ScheduledAction.id == action_id, ScheduledAction.state == "pending")
            .values(state="cancelled")
        )
        await self.session.flush()

    async def count_pending(
        self,
        conversation_id: Optional[int] = None,
        action_type: Optional[str] = None,
    ) -> int:
        stmt = select(ScheduledAction).where(ScheduledAction.state == "pending")
        if conversation_id is not None:
            stmt = stmt.where(ScheduledAction.conversation_id == conversation_id)
        if action_type is not None:
            stmt = stmt.where(ScheduledAction.action_type == action_type)
        result = await self.session.execute(stmt)
        return len(result.scalars().all())

    async def prune(self, older_than_hours: int = 72):
        cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)
        stmt = select(ScheduledAction).where(
            ScheduledAction.state.in_(["executed", "cancelled"]),
            ScheduledAction.executed_at < cutoff,
        )
        result = await self.session.execute(stmt)
        for action in result.scalars().all():
            await self.session.delete(action)
        await self.session.flush()