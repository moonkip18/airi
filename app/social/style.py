"""State-driven typing style.

A person's typing drifts with their state: hyped people caps-emphasize and
get sloppy, irritated people suddenly rediscover punctuation, bored people
type three letters. Her formatting must inherit that drift, or the constant
perfection of every message becomes the AI tell.

Rules of the road:
  * Randomness only ever picks between PLAUSIBLE modes — the modes and their
    probabilities are derived from her emotional state, never uniform dice.
  * Negative states dominate: irritation/embarrassment change typing more
    than positive ones do, exactly like real people.
  * Everything here is guidance for the LLM (how text is WRITTEN) plus
    weights for streaks; the send-side imperfections (typos) live in
    app/telegram/bot.py and read her state at send time.
"""
import random
from typing import Dict

# Streak modes: see ConversationStateManager.maybe_roll_typing_register.
REGISTER_MODES = ("talkative", "normal", "low_effort")


def typing_style_note(emo: Dict[str, float]) -> str:
    """A short prompt block describing how she types RIGHT NOW.

    Returns "" for the neutral case (the system prompt's baseline already
    covers it) — the block only appears when her state actually colors her
    typing, so the prompt itself stays varied rather than carrying another
    permanent instruction.
    """
    annoyance = emo.get("annoyance", 0.0)
    embarrassment = emo.get("embarrassment", 0.0)
    boredom = emo.get("boredom", 0.2)
    energy = emo.get("energy", 0.5)
    amusement = emo.get("amusement", 0.0)

    # Priority: negative states first — they dominate human typing.
    if annoyance >= 0.5:
        return (
            "TYPING RIGHT NOW: irritated. Your punctuation comes back when "
            "you're annoyed ('fine.', 'whatever.'), replies get clipped and "
            "colder, and jokes dry up. Show it, never announce it."
        )
    if embarrassment >= 0.5:
        return (
            "TYPING RIGHT NOW: flustered. Sloppier typing, some hedging, "
            "quick dry-humor recovery. Show it, never announce it."
        )
    if boredom >= 0.55 or energy <= 0.28:
        return (
            "TYPING RIGHT NOW: running on fumes. Dropped words, ultra-short "
            "replies ('idk', 'yeah', 'maybe'), minimal effort — not hostile, "
            "just low battery."
        )
    if (amusement >= 0.55 and energy >= 0.55) or energy >= 0.78:
        return (
            "TYPING RIGHT NOW: hyped. Faster, lazier typing, occasional CAPS "
            "on one word for emphasis ('that's SO good'), hotter reactions. "
            "Still natural — never performative."
        )
    return ""


def register_weights(emo: Dict[str, float]) -> Dict[str, float]:
    """Relative weights for the verbosity streak modes.

    State tilts the distribution; the dice only choose within it. High
    energy/amusement/interest makes a chatty stretch more likely, boredom
    and low energy make a low-effort stretch more likely.
    """
    energy = emo.get("energy", 0.5)
    amusement = emo.get("amusement", 0.0)
    boredom = emo.get("boredom", 0.2)
    annoyance = emo.get("annoyance", 0.0)
    interest = emo.get("interest", 0.5)
    social = emo.get("social_engagement", 0.5)

    weights = {"talkative": 0.25, "normal": 0.55, "low_effort": 0.20}

    if energy >= 0.65 or amusement >= 0.55:
        weights["talkative"] *= 1.8
    if interest >= 0.6 or social >= 0.65:
        weights["talkative"] *= 1.3
    if boredom >= 0.5 or energy <= 0.35:
        weights["low_effort"] *= 2.0
        weights["talkative"] *= 0.5
    if annoyance >= 0.5:
        weights["low_effort"] *= 1.5

    total = sum(weights.values())
    if total <= 0:
        return {"talkative": 0.25, "normal": 0.55, "low_effort": 0.20}
    return {k: v / total for k, v in weights.items()}


def roll_register(emo: Dict[str, float], rng=random) -> str:
    """Pick a verbosity mode, weighted by her current state."""
    weights = register_weights(emo)
    return rng.choices(
        list(weights.keys()), weights=list(weights.values()), k=1,
    )[0]


def register_note(register: str) -> str:
    """Prompt block for the current verbosity streak ("").

    "normal" is the baseline and emits nothing — only the streaks that
    actually change her output are worth prompt tokens.
    """
    if register == "talkative":
        return (
            "TYPING MODE: chatty stretch. You're on a roll right now — "
            "fuller replies (1-3 sentences) feel natural, you build on your "
            "own points."
        )
    if register == "low_effort":
        return (
            "TYPING MODE: low-effort stretch. You're replying from the couch "
            "— a few words max, fragments, no elaboration unless directly "
            "asked."
        )
    return ""
