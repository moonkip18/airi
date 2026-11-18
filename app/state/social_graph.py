"""Lightweight social graph.

Airi observes not just her own relationships but relationships *between other
people* (e.g. "Alex and Sarah seem close", "Bob annoys everyone"). A small
undirected edge table is sufficient — no graph database.

Edges are updated gradually from observations and are used to enrich the
relationship model (e.g. jealousy triggers, closeness inference).
"""
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.database import SocialGraphEdge

logger = structlog.get_logger()


class SocialGraphManager:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_edge(self, user_a_id: int, user_b_id: int) -> Optional[SocialGraphEdge]:
        if user_a_id == user_b_id:
            return None
        low, high = sorted((user_a_id, user_b_id))
        stmt = select(SocialGraphEdge).where(
            SocialGraphEdge.user_a_id == low,
            SocialGraphEdge.user_b_id == high,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def observe_interaction(
        self,
        user_a_id: int,
        user_b_id: int,
        warmth: float = 0.0,       # -1..1 observed warmth vs antagonism
        intensity: float = 0.5,
    ) -> Optional[SocialGraphEdge]:
        """A conversation event between two users nudges their apparent bond."""
        if user_a_id is None or user_b_id is None or user_a_id == user_b_id:
            return None

        edge = await self.get_edge(user_a_id, user_b_id)
        if edge is None:
            edge = SocialGraphEdge(
                user_a_id=min(user_a_id, user_b_id),
                user_b_id=max(user_a_id, user_b_id),
                closeness=0.1,
                polarity=0.0,
                interaction_count=0,
            )
            self.session.add(edge)

        # Gradual: one observation moves the score by a fraction.
        delta = 0.05 * intensity
        edge.polarity = max(-1.0, min(1.0, edge.polarity + warmth * delta * 4))
        edge.closeness = max(0.0, min(1.0, edge.closeness + (0.2 if abs(warmth) < 0.3 else 0.15) * delta))
        edge.interaction_count += 1
        edge.last_seen_at = datetime.utcnow()
        edge.updated_at = datetime.utcnow()
        await self.session.flush()
        return edge

    async def get_edges_for_user(self, user_id: int) -> List[SocialGraphEdge]:
        stmt = select(SocialGraphEdge).where(
            or_(
                SocialGraphEdge.user_a_id == user_id,
                SocialGraphEdge.user_b_id == user_id,
            )
        ).order_by(SocialGraphEdge.interaction_count.desc()).limit(20)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_edge_summaries(self, user_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Map user_id -> summary of that user's observed social bonds."""
        summaries: Dict[int, Dict[str, Any]] = {}
        for uid in user_ids:
            edges = await self.get_edges_for_user(uid)
            others = []
            for e in edges:
                other = e.user_b_id if e.user_a_id == uid else e.user_a_id
                others.append({
                    "user_id": other,
                    "closeness": e.closeness,
                    "polarity": e.polarity,
                })
            summaries[uid] = {"bonds": others}
        return summaries

    async def closeness_between(self, user_a_id: int, user_b_id: int) -> float:
        edge = await self.get_edge(user_a_id, user_b_id)
        if edge is None:
            return 0.0
        return edge.closeness

    async def most_connected(self, user_ids: List[int], limit: int = 5) -> List[int]:
        """Users who are most socially connected (center of the group)."""
        if not user_ids:
            return []
        stmt = select(SocialGraphEdge)
        result = await self.session.execute(stmt)
        degree: Dict[int, int] = {uid: 0 for uid in user_ids}
        for e in result.scalars().all():
            if e.user_a_id in degree:
                degree[e.user_a_id] += e.interaction_count
            if e.user_b_id in degree:
                degree[e.user_b_id] += e.interaction_count
        return sorted(degree, key=degree.get, reverse=True)[:limit]