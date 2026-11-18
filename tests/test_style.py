"""State-driven typing style tests (app/social/style.py + wiring)."""
import random

import pytest

from app.social.style import (
    REGISTER_MODES,
    register_weights,
    roll_register,
    typing_style_note,
)


# ---------------------------------------------------------------------------
# Mood -> typing style note
# ---------------------------------------------------------------------------

NEUTRAL = {
    "mood": 0.0, "energy": 0.5, "curiosity": 0.5, "amusement": 0.0,
    "annoyance": 0.0, "interest": 0.5, "social_engagement": 0.5,
    "confidence": 0.5, "embarrassment": 0.0, "competitiveness": 0.5,
    "boredom": 0.2,
}


def _emo(**overrides):
    emo = dict(NEUTRAL)
    emo.update(overrides)
    return emo


def test_neutral_state_has_no_style_note():
    assert typing_style_note(_emo()) == ""


def test_annoyance_brings_back_punctuation():
    note = typing_style_note(_emo(annoyance=0.7))
    assert "irritated" in note
    assert "punctuation" in note


def test_annoyance_dominates_amusement():
    note = typing_style_note(_emo(annoyance=0.8, amusement=0.9, energy=0.9))
    assert "irritated" in note


def test_hyped_state_note():
    note = typing_style_note(_emo(amusement=0.7, energy=0.7))
    assert "hyped" in note


def test_low_energy_note():
    note = typing_style_note(_emo(energy=0.15))
    assert "fumes" in note or "low energy" in note


def test_boredom_triggers_low_energy_note():
    note = typing_style_note(_emo(boredom=0.7))
    assert "fumes" in note or "low energy" in note


def test_embarrassment_note():
    note = typing_style_note(_emo(embarrassment=0.7))
    assert "flustered" in note


# ---------------------------------------------------------------------------
# Verbosity streak weights
# ---------------------------------------------------------------------------

def test_weights_are_normalized():
    for emo in (
        _emo(),
        _emo(energy=0.9, amusement=0.8),
        _emo(boredom=0.8, energy=0.2),
        _emo(annoyance=0.7),
    ):
        weights = register_weights(emo)
        assert set(weights.keys()) == set(REGISTER_MODES)
        assert sum(weights.values()) == pytest.approx(1.0)


def test_boredom_pushes_toward_low_effort():
    bored = register_weights(_emo(boredom=0.8, energy=0.3))
    lively = register_weights(_emo(energy=0.9, amusement=0.8))
    assert bored["low_effort"] > lively["low_effort"]
    assert lively["talkative"] > bored["talkative"]


def test_high_energy_pushes_toward_talkative():
    lively = register_weights(_emo(energy=0.9, amusement=0.8, interest=0.8))
    assert lively["talkative"] > lively["low_effort"]


def test_roll_register_returns_valid_mode():
    rng = random.Random(42)
    for _ in range(50):
        assert roll_register(_emo(), rng=rng) in REGISTER_MODES


def test_roll_register_respects_state_skew():
    rng = random.Random(7)
    emo = _emo(boredom=0.9, energy=0.1)
    modes_list = [roll_register(emo, rng=rng) for _ in range(300)]
    # Heavy boredom + low energy: low_effort should clearly beat talkative.
    assert modes_list.count("low_effort") > modes_list.count("talkative") * 2


# ---------------------------------------------------------------------------
# SocialDecision.double_text
# ---------------------------------------------------------------------------

def test_social_decision_double_text_defaults_empty():
    from app.social.decision import SocialAction, SocialDecision

    decision = SocialDecision(
        action=SocialAction.REPLY,
        reasoning_summary="test",
        current_topic="games",
        confidence=0.8,
        response="yeah",
    )
    assert decision.double_text == ""


# ---------------------------------------------------------------------------
# Send-side imperfections
# ---------------------------------------------------------------------------

def test_make_typo_sometimes_produces_a_typo():
    from app.telegram.bot import TelegramBot

    text = "this is definitely long enough to produce typos sometimes"
    changed = 0
    for _ in range(60):
        result = TelegramBot._make_typo(text)
        if result is not None and result != text:
            changed += 1
    assert changed > 0


def test_make_typo_never_returns_garbage():
    from app.telegram.bot import TelegramBot

    text = "completely different words here okay"
    made_any = False
    for _ in range(40):
        result = TelegramBot._make_typo(text)
        if result is None:
            continue
        made_any = True
        # Same length overall, same word count, and every word is either
        # untouched or an adjacent-letter swap of the original.
        assert len(result) == len(text)
        orig_words, new_words = text.split(), result.split()
        assert len(orig_words) == len(new_words)
        for ow, nw in zip(orig_words, new_words):
            if ow == nw:
                continue
            assert len(ow) == len(nw)
            diffs = [i for i, (a, b) in enumerate(zip(ow, nw)) if a != b]
            # An adjacent swap touches exactly 2 neighboring positions.
            assert diffs == [0, 1] or all(
                diffs[i + 1] - diffs[i] == 1 for i in range(len(diffs) - 1)
            )
    assert made_any
