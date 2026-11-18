"""Profanity filter tests (app/social/censor.py)."""
import pytest

from app.config import settings
from app.social.censor import censor


def test_hard_profanity_is_filtered():
    assert censor("what the fuck") == "what the [filtered]"
    assert censor("this game is shit") == "this game is [filtered]"
    assert censor("you bitch") == "you [filtered]"


def test_conjugations_are_filtered():
    assert "fucking" not in censor("that's fucking sick")
    assert censor("that's fucking sick") == "that's [filtered] sick"
    assert censor("bitches be crazy") == "[filtered] be crazy"


def test_case_insensitive():
    assert censor("FUCK off") == "[filtered] off"
    assert censor("What the FUCK") == "What the [filtered]"


def test_multiple_words_filtered():
    assert censor("shit. fucking hell") == "[filtered]. [filtered] hell"


def test_mild_swear_words_pass_through():
    text = "damn, hell, crap, ass, piss off"
    assert censor(text) == text


def test_word_boundaries_protect_innocent_words():
    assert censor("cocktail hour") == "cocktail hour"
    assert censor("that class was long") == "that class was long"
    assert censor("Charles Dickens wrote") == "Charles Dickens wrote"
    assert censor("shiitake mushroom") == "shiitake mushroom"


def test_stretched_letters_still_filtered():
    # f+u+c+k+ pattern catches "fuuuck"-style emphasis.
    assert censor("fuuuck offf") == "[filtered] offf"

def test_empty_and_none_ish_input():
    assert censor("") == ""
    assert censor("clean text") == "clean text"


def test_filter_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "profanity_filter", False)
    assert censor("what the fuck") == "what the fuck"
