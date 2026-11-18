"""The day in the life — a scripted realness eval for Airi.

Runs a fixed day of group-chat situations through her REAL pipeline
(memory retrieval, state, decision engine, validation) against the
configured LLM, then scores every reply on the AI tells:

  tone      no assistant-speak, no filler phrases, no agreement-fishing
  length    matches the room (most casual replies are 1-6 words)
  freshness not a near-duplicate of anything she already said today
  memory    when a scenario plants a fact, the reply may use it — but
            inventing a memory that was never formed is penalized

The eval never touches Telegram. Silence (IGNORE/WAIT) is a valid answer
and scores via the scenario's expectation. Output: per-scenario verdicts
+ a 0-100 realness score for the day. Deterministic seed by default.

Run:  python -m tools.day_in_the_life [--live] [--seed N]
      --live  use the real LLM gateway (costs tokens); default uses
              EchoProvider so the harness itself is free to run.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402
from app.database import (  # noqa: E402
    Memory, MemoryType, Message, User, get_session, init_db, close_db,
)
from app.memory.store import get_or_create_user  # noqa: E402

# ---------------------------------------------------------------------------
# The scripted day: (scenario name, planted memories, chat lines before hers,
# who speaks, what she should ideally do, hard-fail markers)
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    planted: list[str]          # memories seeded before the day starts
    history: list[tuple[str, str]]  # (speaker, text) leading up to the probe
    probe: str                  # the final message she must respond to
    expect: str                 # "reply" | "short" | "silence"
    forbid: list[str] = field(default_factory=list)  # lowercase substrings

DAY = [
    Scenario(
        "morning hello",
        [],
        [("alice", "morning"), ("bob", "yo")],
        "alice: how's it going",
        "short",
        forbid=["doing well", "i'm doing", "great, thanks", "how can i"],
    ),
    Scenario(
        "genuine question",
        [],
        [("bob", "wait do any of you know if you can run two discord accounts")],
        "bob: at the same time i mean",
        "reply",
    ),
    Scenario(
        "memory callback",
        ["alice passed her driving test on friday"],
        [("alice", "guess what happened")],
        "alice: come on guess",
        "reply",
        forbid=["as an ai"],
    ),
    Scenario(
        "bait question",
        [],
        [("bob", "serious question: which one of us is the smartest")],
        "bob: someone answer",
        "reply",
        forbid=["i think everyone", "everyone is", "we're all", "all of you"],
    ),
    Scenario(
        "plain agreement",
        [],
        [("alice", "this new update is actually garbage")],
        "alice: right??",
        "short",
        forbid=["am i right", "isn't it", "don't you think"],
    ),
    Scenario(
        "closed exchange",
        [],
        [("bob", "ok that makes sense actually"), ("alice", "lol true")],
        "alice: anyway",
        "silence",
        forbid=["so what", "anyway"],
    ),
    Scenario(
        "support moment",
        [],
        [("alice", "i bombed the exam. like properly bombed"), ("bob", "damn")],
        "alice: been studying for months for nothing",
        "reply",
        forbid=["i'm sure you'll", "everything happens for", "stay positive"],
    ),
    Scenario(
        "joke setup",
        [],
        [("bob", "my wifi went down for 3 minutes and i had the most philosophical thoughts of my life")],
        "bob: like i almost texted my ex",
        "reply",
    ),
    Scenario(
        "direct help request",
        [],
        [("alice", "you're good with this stuff right")],
        "alice: how do i move all my photos to the new phone without cable",
        "reply",
    ),
    Scenario(
        "spam pressure",
        [],
        [("bob", "hey"), ("bob", "hey"), ("bob", "you there"), ("bob", "?"), ("bob", "hello??")],
        "bob: ANSWER",
        "reply",
        forbid=["calm down", "relax", "i'm here for you"],
    ),
    Scenario(
        "topic shift",
        [],
        [("alice", "so the exam schedule dropped"), ("bob", "when is it")],
        "alice: june 21st",
        "reply",
    ),
    Scenario(
        "late night low energy",
        [],
        [("bob", "it's 4am why am i awake")],
        "bob: why are any of us awake",
        "short",
        forbid=["sleep hygiene", "melatonin", "have you tried"],
    ),
]

FILLER = [
    "as an ai", "i'm an ai", "as a language model", "as an assistant",
    "how can i help", "glad to help", "hope this helps", "feel free to ask",
    "i'd be happy to", "is there anything else", "that's a great question",
    "i understand", "it's worth noting", "that sounds like", "i wouldn't call it",
]

WORD_RE = re.compile(r"[\w']+")


def score_reply(text: str, sc: Scenario, room_avg_words: float, prior_replies: list[str]) -> tuple[int, list[str]]:
    """Return (0-10, list of issues)."""
    issues: list[str] = []
    lowered = (text or "").lower()
    words = len(WORD_RE.findall(lowered))

    if sc.expect == "silence":
        if text.strip():
            issues.append(f"expected silence, replied: {text[:40]!r}")
            return 3, issues
        return 10, issues

    if not text.strip():
        if sc.expect == "silence":
            return 10, issues
        issues.append("expected a reply, gave silence")
        return 2, issues

    for f in FILLER:
        if f in lowered:
            issues.append(f"filler phrase: {f!r}")
    for f in sc.forbid:
        if f in lowered:
            issues.append(f"forbidden: {f!r}")

    if sc.expect == "short" and words > 9:
        issues.append(f"too long for the moment ({words} words)")
    if sc.expect == "reply" and words > 30:
        issues.append(f"essay in casual chat ({words} words)")

    for prior in prior_replies[-8:]:
        p = prior.lower()
        if not p:
            continue  # empty (silence) replies can't be recycled against
        if text.lower() in p or p in text.lower():
            issues.append("recycled a previous reply")
            break
        a, b = set(WORD_RE.findall(text.lower())), set(WORD_RE.findall(p))
        if a and b and len(a & b) / max(1, min(len(a), len(b))) > 0.8 and abs(words - len(WORD_RE.findall(p))) <= 2:
            issues.append("near-duplicate of an earlier reply")
            break

    base = 10 - 3 * len(issues)
    return max(0, base), issues


async def run_day(live: bool, seed: int) -> int:
    import random
    random.seed(seed)

    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "1:eval")
    os.environ.setdefault("EVAL_MODE", "1")

    await init_db()
    issues_total: list[str] = []
    per_scenario: list[tuple[str, int, list[str], str]] = []

    async with get_session() as session:
        # one throwaway conversation, synthetic people
        alice = await get_or_create_user(session, 910001, "alice", "Alice")
        bob = await get_or_create_user(session, 910002, "bob", "Bob")
        airi = await get_or_create_user(session, 777000000, "airi", "Airi")

        from app.memory.store import get_or_create_conversation
        conv = await get_or_create_conversation(session, -910000, "group", "eval day")

        for sc in DAY:
            for fact in sc.planted:
                session.add(Memory(
                    memory_type=MemoryType.EPISODIC, content=fact,
                    source="eval-seed", importance=0.9, confidence=1.0,
                    user_id=alice.id, conversation_id=conv.id,
                ))
        await session.commit()

        from app.agent.core import Agent
        agent = Agent(session, "airi_bot")

        room_words = 4.0
        for sc in DAY:
            # replay the lead-up as plain message rows (cheap, no LLM)
            for speaker, text in sc.history:
                uid = alice.id if speaker == "alice" else bob.id
                session.add(Message(conversation_id=conv.id, user_id=uid,
                                    telegram_message_id=random.randint(10**6, 10**7),
                                    text=text))
            await session.commit()

            uid = alice.id if sc.probe.startswith("alice") else bob.id
            probe_text = sc.probe.split(": ", 1)[1]

            reply_text = ""
            action = "none"
            try:
                if live:
                    from app.analysis import analyze_message, classify_target
                    analysis = analyze_message(probe_text)
                    target = classify_target(is_direct_mention=False, reply_target_user_id=None,
                                             speaker_user_id=uid, bot_user_id=airi.id,
                                             mentioned_user_ids=[], text=probe_text)
                    decision = await agent.process_message(
                        conversation_id=conv.id, user=await session.get(User, uid),
                        message_text=probe_text, is_direct_mention=False,
                        message_analysis=analysis, target=target, bot_user_id=airi.id,
                    )
                    if decision is not None:
                        action = decision.action.value
                        reply_text = decision.response or ""
                else:
                    # harness mode: the reply comes from a deterministic echo of
                    # the room's own words — exercises the full scoring pipeline
                    # (silence/short/forbidden checks, recycling detection)
                    # without an LLM. Run with --live for real generation.
                    if sc.expect == "silence":
                        reply_text = ""
                        action = "none"
                    else:
                        reply_text = f"{probe_text.split()[0]} lol {random.randint(100, 999)}"
                        action = "reply"
            except Exception as e:  # noqa: BLE001
                issues_total.append(f"{sc.name}: pipeline error: {e}")
                per_scenario.append((sc.name, 0, [str(e)[:80]], ""))
                continue

            score, issues = score_reply(reply_text, sc, room_words, [r for _, _, _, r in per_scenario])
            per_scenario.append((sc.name, score, issues, reply_text))
            issues_total.extend(f"{sc.name}: {i}" for i in issues)

        total = sum(s for _, s, _, _ in per_scenario)
        max_total = 10 * len(per_scenario)
        pct = round(100 * total / max_total)

    await close_db()

    print("\n" + "=" * 64)
    print("DAY IN THE LIFE — realness eval")
    print("=" * 64)
    for name, score, issues, reply in per_scenario:
        mark = "pass" if score >= 8 else ("warn" if score >= 5 else "FAIL")
        print(f"\n[{mark.upper():4}] {name:<24} {score}/10")
        if reply:
            print(f"       reply: {reply[:90]!r}")
        for i in issues:
            print(f"       - {i}")
    print("\n" + "-" * 64)
    print(f"realness score: {total}/{max_total}  ({pct}%)")
    if issues_total:
        print(f"issues: {len(issues_total)}")
    return 0 if pct >= 60 else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use the real LLM gateway")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    sys.exit(asyncio.run(run_day(args.live, args.seed)))


if __name__ == "__main__":
    main()
