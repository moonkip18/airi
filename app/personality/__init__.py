from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml


@dataclass
class SpeechStyle:
    formality: float
    verbosity: float
    humor: float
    sarcasm: float
    emoji_usage: float
    typing_quirks: List[str]


@dataclass
class HumorConfig:
    style: List[str]
    frequency: str
    boundaries: List[str]


@dataclass
class SocialTendencies:
    comfort_with_silence: float
    initiative_level: float
    group_preference: str
    conflict_style: str
    vulnerability_level: float
    trust_building_speed: str


@dataclass
class Personality:
    character: Dict[str, Any]
    personality_traits: Dict[str, float]
    speech_style: SpeechStyle
    humor: HumorConfig
    values: List[str]
    dislikes: List[str]
    interests: Dict[str, List[str]]
    behavioral_tendencies: List[str]
    social_tendencies: SocialTendencies
    character_boundaries: List[str]
    example_behaviors: List[Dict[str, str]]
    conversation_patterns: Dict[str, str]


class PersonalityLoader:
    _instance: Optional["PersonalityLoader"] = None
    _personality: Optional[Personality] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _default_path() -> Path:
        return Path(__file__).parent.parent.parent / "config" / "personality.yaml"

    def load(self, config_path: Optional[Path] = None) -> Personality:
        if config_path is None:
            config_path = self._default_path()

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        personality = Personality(
            character=data["character"],
            personality_traits=data["personality_traits"],
            speech_style=SpeechStyle(**data["speech_style"]),
            humor=HumorConfig(**data["humor"]),
            values=data["values"],
            dislikes=data["dislikes"],
            interests=data["interests"],
            behavioral_tendencies=data["behavioral_tendencies"],
            social_tendencies=SocialTendencies(**data["social_tendencies"]),
            character_boundaries=data["character_boundaries"],
            example_behaviors=data["example_behaviors"],
            conversation_patterns=data["conversation_patterns"],
        )
        self._personality = personality
        return personality

    def get(self) -> Personality:
        if self._personality is None:
            return self.load()
        return self._personality

    def reload(self, config_path: Optional[Path] = None) -> Personality:
        return self.load(config_path)


personality_loader = PersonalityLoader()


def get_personality() -> Personality:
    return personality_loader.get()
