"""Smoke test: boot the dashboard app against a temp DB with synthetic data,
hit every endpoint, verify shapes. No real data, no LLM, no Telegram.
Run from the airi publish root:  python tools/test_dashboard.py
"""
import asyncio
import os
import sys
import tempfile

os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/dash_test.db"
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "1:test")
os.environ.setdefault("LLM_API_KEY", "test")
os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("LLM_MODEL", "test-model")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta

from httpx import ASGITransport, AsyncClient


async def main() -> int:
    from app.database import (
        EmotionalState, LanguageStats, Memory, MemoryType, Message,
        Relationship, TrainingSample, User, get_session, init_db, close_db,
    )
    from app.dashboard import app

    await init_db()
    now = datetime.utcnow()
    async with get_session() as s:
        u1 = User(telegram_id=100, username="alice", first_name="Alice")
        u2 = User(telegram_id=101, username="bob", first_name="Bob")
        bot = User(telegram_id=999, username="airi_bot", first_name="Airi")
        s.add_all([u1, u2, bot])
        await s.flush()
        s.add_all([
            Relationship(user_id=u1.id, affection=0.4, trust=0.6, annoyance=0.1,
                         familiarity=0.7, interaction_count=42,
                         last_interaction=now),
            Relationship(user_id=u2.id, affection=-0.2, trust=0.1, annoyance=0.5,
                         familiarity=0.3, interaction_count=7),
        ])
        s.add_all([
            Memory(memory_type=MemoryType.EPISODIC, content="alice passed her exam",
                   source="extraction", importance=0.8, confidence=0.9),
            Memory(memory_type=MemoryType.SEMANTIC, content="bob dislikes mondays",
                   source="extraction", importance=0.4, confidence=0.7),
        ])
        for i in range(10):
            s.add(Message(conversation_id=1,
                          user_id=bot.id if i % 2 else u1.id,
                          telegram_message_id=i, text=f"msg {i}",
                          is_from_bot=(i % 2 == 1),
                          timestamp=now - timedelta(days=i % 5)))
        s.add(TrainingSample(trigger="reply", response_text="lol yeah",
                             action="reply", language="english"))
        s.add(EmotionalState(mood=0.2, energy=0.7, boredom=0.1))
        s.add(LanguageStats(conversation_id=1, english_responses=9,
                            persian_responses=1, persian_snippets=2))
        await s.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/")
        assert r.status_code == 200 and "airi" in r.text, "index failed"
        print("GET /                        OK")

        r = await client.get("/api/health")
        assert r.status_code == 200 and r.json()["ok"], "health failed"
        print("GET /api/health              OK")

        r = await client.get("/api/memories")
        body = r.json()
        assert r.status_code == 200 and body["total"] == 2, f"memories: {body}"
        assert body["rows"][0]["type"] in ("episodic", "semantic")
        print("GET /api/memories            OK")

        r = await client.get("/api/memories?q=exam")
        assert r.json()["total"] == 1, "search failed"
        r = await client.get("/api/memories?type=semantic")
        assert r.json()["total"] == 1, "type filter failed"
        print("GET /api/memories?q=&type=   OK")

        mem_id = body["rows"][0]["id"]
        r = await client.patch(f"/api/memories/{mem_id}",
                               json={"content": "alice passed her retake"})
        assert r.status_code == 200 and r.json()["ok"], "patch failed"
        print(f"PATCH /api/memories/{mem_id}      OK")

        r = await client.delete(f"/api/memories/{mem_id}")
        assert r.status_code == 200, "delete failed"
        r = await client.get("/api/memories")
        assert r.json()["total"] == 1, "delete did not persist"
        print(f"DELETE /api/memories/{mem_id}     OK")

        r = await client.get("/api/people")
        ppl = r.json()["rows"]
        assert r.status_code == 200 and len(ppl) == 2, "people failed"
        assert ppl[0]["name"] == "alice" and ppl[0]["interactions"] == 42
        print("GET /api/people              OK")

        r = await client.get("/api/metrics")
        m = r.json()
        assert r.status_code == 200, "metrics failed"
        assert m["totals"]["messages"] == 10, f"messages: {m['totals']}"
        assert m["totals"]["people"] == 3
        assert sum(m["language"].values()) == 12
        assert m["emotional_state"]["mood"] == 0.2
        assert any(p["bot"] > 0 for p in m["per_day"]), "bot split missing"
        print("GET /api/metrics             OK")

    await close_db()
    print("\nall dashboard smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
