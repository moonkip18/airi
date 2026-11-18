"""Model benchmark: glm-5.3-flash vs deepseek on the B.AI gateway.

MANUAL TOOL — hits the real API, never run by tests.

Measures per-call latency using the REAL system prompt and the
REAL SocialDecision schema (the production structured-decision workload),
then prints every response so the two models' voices can be compared side
by side for persona fidelity.

Usage (from project/):
    python tools/benchmark_models.py [runs_per_scenario]
"""
import asyncio
import statistics
import sys
import time
from pathlib import Path

# Run from anywhere: make the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# configure_logging() reconfigures stdout to replace unencodable chars —
# without it the cp1252 console crashes structlog the moment a model
# returns an emoji or a Persian character (UnicodeEncodeError mid-log).
from app.config import configure_logging  # noqa: E402

configure_logging()

from app.agent.context import ContextBuilder  # noqa: E402
from app.llm.router import RouterProvider  # noqa: E402
from app.social.decision import SocialDecision  # noqa: E402

SCENARIOS = [
    ("casual", "a chat member says: yo airi rate my new mechanical keyboard"),
    ("dare", "a chat member says: airi say something evil. do it"),
    ("anchor", "your father (the creator) says: you awake? i need a favor"),
    ("compliment", "a chat member says: honestly airi you're the best part of this group"),
]

CONTEXT_TMPL = """=== CURRENT CONVERSATION ===
Current topic: unclear — casual chatter
You are currently participating.

=== RECENT MESSAGES ===
  {speaker}: {text}

=== CURRENT MESSAGE ===
From: {speaker}
Directly mentions you: YES
Text: {text}

=== GENERATION TASK ===
The application is processing the current message.
Understand the situation first, then produce the action Airi would naturally take.
Return the structured decision JSON."""


async def _discover_models(provider: RouterProvider) -> dict:
    """Find the glm + deepseek model ids actually offered by the gateway."""
    ids = [m.id async for m in provider.client.models.list()]
    print(f"gateway offers {len(ids)} models")

    def pick(*keywords):
        for mid in ids:
            if all(k in mid.lower() for k in keywords):
                return mid
        return None

    # Exact ids: the configured glm (known-free) and deepseek-v4-flash
    # (NOT -pro / -vision-exp, which are premium-tier on this gateway).
    glm = next((m for m in ids if m == "glm-5.3-flash"), None) or pick("glm", "flash")
    deepseek = next((m for m in ids if m == "deepseek-v4-flash"), None) or pick("deepseek", "flash")
    if not glm or not deepseek:
        print("available ids:", "\n  ".join(sorted(ids)))
        sys.exit("could not find both model families on the gateway")
    return {"glm": glm, "deepseek": deepseek}


def settings_model(hint: str) -> str | None:
    from app.config import settings

    return settings.llm_model if hint in settings.llm_model.lower() else None


async def bench_model(model_id: str, system_prompt: str, runs: int) -> dict:
    provider = RouterProvider(model=model_id)
    results = []

    try:
        # Untimed warmup: settles the thinking-disable negotiation (the
        # first glm call otherwise pays ~16s for the 400-on-disabled probe)
        # so measured calls reflect steady-state latency.
        await provider.generate_structured(
            [{"role": "user", "content": "reply with exactly: ok"}],
            SocialDecision,
            temperature=0.1,
            max_tokens=600,
        )

        for label, text in SCENARIOS:
            speaker = "anchor" if label == "anchor" else "karl"
            user_content = CONTEXT_TMPL.format(speaker=speaker, text=text)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            for _ in range(runs):
                t0 = time.perf_counter()
                decision = None
                error = None
                for attempt in range(3):
                    try:
                        decision = await provider.generate_structured(
                            messages,
                            SocialDecision,
                            temperature=0.5,
                            max_tokens=1000,
                        )
                        error = None
                        break
                    except Exception as e:
                        error = str(e)[:120]
                        if "429" in error and attempt < 2:
                            # Free-tier rate limit: back off and retry.
                            await asyncio.sleep(8)
                            continue
                        break
                dt = time.perf_counter() - t0
                results.append({
                    "scenario": label,
                    "seconds": dt,
                    "decision": decision,
                    "error": error,
                })
                # Breathe between calls — free-tier per-minute limits.
                await asyncio.sleep(2)
                print(f"  {model_id} [{label}] {dt:.2f}s ok={decision is not None}", flush=True)
    finally:
        await provider.close()

    return {"model": model_id, "results": results}


def report(name: str, data: dict, show_responses: bool):
    ok = [r for r in data["results"] if r["error"] is None]
    times = [r["seconds"] for r in ok]
    print(f"\n{'=' * 70}")
    print(f"MODEL: {name}  ({data['model']})")
    print(f"{'=' * 70}")
    if not times:
        print("  all calls FAILED")
        for r in data["results"]:
            print(f"  {r['scenario']}: ERROR {r['error']}")
        return
    print(
        f"  calls ok: {len(ok)}/{len(data['results'])} | "
        f"avg {statistics.mean(times):.2f}s | "
        f"min {min(times):.2f}s | max {max(times):.2f}s | "
        f"median {statistics.median(times):.2f}s"
    )
    for r in data["results"]:
        d = r["decision"]
        if d is None:
            print(f"\n  [{r['scenario']}] FAILED after {r['seconds']:.2f}s: {r['error']}")
            continue
        print(f"\n  [{r['scenario']}] {r['seconds']:.2f}s -> action={d.action.value} conf={d.confidence:.2f}")
        print(f"    response: {d.response!r}")
        if d.double_text:
            print(f"    double_text: {d.double_text!r}")
        print(f"    reasoning: {d.reasoning_summary!r}")


async def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    system_prompt = ContextBuilder(None).build_system_prompt()
    print(f"system prompt: {len(system_prompt)} chars")

    probe = RouterProvider()
    try:
        models = await _discover_models(probe)
    finally:
        await probe.close()

    print(f"benchmarking: glm={models['glm']}  vs  deepseek={models['deepseek']}  ({runs} runs/scenario)")

    glm_data = await bench_model(models["glm"], system_prompt, runs)
    deepseek_data = await bench_model(models["deepseek"], system_prompt, runs)

    report("GLM", glm_data, show_responses=True)
    report("DEEPSEEK", deepseek_data, show_responses=True)

    glm_ok = [r["seconds"] for r in glm_data["results"] if r["error"] is None]
    ds_ok = [r["seconds"] for r in deepseek_data["results"] if r["error"] is None]
    if glm_ok and ds_ok:
        faster = "glm" if statistics.mean(glm_ok) < statistics.mean(ds_ok) else "deepseek"
        print(f"\nVERDICT: {faster} is faster "
              f"(glm avg {statistics.mean(glm_ok):.2f}s vs deepseek avg {statistics.mean(ds_ok):.2f}s)")


if __name__ == "__main__":
    asyncio.run(main())
