"""Curate training samples for the distillation loop.

Distillation-loop step 2: human curation. Shows unreviewed training samples
(conversation context + her actual reply); you judge whether the reply is
on-voice. Samples rated `keep` become few-shot voice exemplars in her
system prompt (app/agent/context.py load_exemplars).

Usage:
    python tools/rate_samples.py [max_samples] [--oldest]

    --oldest   review the OLDEST unrated samples first (the early "real"
               era) instead of newest — useful for the first big pass.

Keys: y = on-voice (rating=1) | n = off-voice (rating=0) | s = skip | q = quit
"""
import asyncio
import sys

from sqlalchemy import select, desc, asc, func

from app.database import TrainingSample, get_session


async def main(limit: int = 20, oldest_first: bool = False):
    async with get_session() as session:
        unrated = await session.scalar(
            select(func.count(TrainingSample.id)).where(
                TrainingSample.rating.is_(None)
            )
        )
        kept = await session.scalar(
            select(func.count(TrainingSample.id)).where(TrainingSample.rating == 1)
        )

        order = asc if oldest_first else desc
        stmt = (
            select(TrainingSample)
            .where(TrainingSample.rating.is_(None))
            .order_by(order(TrainingSample.created_at))
            .limit(limit)
        )
        samples = (await session.execute(stmt)).scalars().all()

        if not samples:
            print("No unreviewed samples. They appear as she chats.")
            return

        print(
            f"unrated total: {unrated} | currently kept as exemplars: {kept}"
            f" | showing {len(samples)} ({'oldest' if oldest_first else 'newest'} first)\n"
        )

        rated = 0
        for i, s in enumerate(samples, 1):
            print("=" * 64)
            print(f"[{i}/{len(samples)}]")
            for m in (s.context_snapshot or [])[-6:]:
                sender = m.get("sender", "?")
                print(f"  {sender}: {m.get('text', '')}")
            print(f"  --- her reply ({s.trigger}) ---")
            print(f"  {s.response_text}")

            while True:
                try:
                    ans = input(
                        "[y] on-voice / [n] off-voice / [s] skip / [q] quit > "
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    ans = "q"
                if ans in ("y", "n", "s", "q"):
                    break
                print("  y / n / s / q")

            if ans == "q":
                break
            if ans in ("y", "n"):
                s.rating = 1 if ans == "y" else 0
                rated += 1

        await session.commit()

        kept_total = await session.scalar(
            select(func.count(TrainingSample.id)).where(TrainingSample.rating == 1)
        )
        print(
            f"\nRated {rated} sample(s) this session. Exemplars now in the "
            f"prompt: {kept_total} (cap 8, newest-kept first)."
        )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    max_samples = int(args[0]) if args else 20
    oldest = "--oldest" in sys.argv
    asyncio.run(main(max_samples, oldest))
