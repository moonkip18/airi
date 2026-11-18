"""Message targeting: who is this message actually aimed at?

Classification prevents Airi from treating every group message as an
invitation to speak. Deterministic, no LLM.
"""
from enum import Enum


class MessageTarget(str, Enum):
    DIRECTLY_TO_AIRI = "directly_to_airi"
    TO_ANOTHER_USER = "to_another_user"
    REPLY_TO_ANOTHER_USER = "reply_to_another_user"
    GENERAL_GROUP_MESSAGE = "general_group_message"
    UNCLEAR = "unclear"


def classify_target(
    *,
    is_direct_mention: bool,
    reply_target_user_id=None,
    speaker_user_id=None,
    bot_user_id=None,
    mentioned_user_ids=None,
    message_type: str = "text",
    text: str = "",
) -> MessageTarget:
    """Classify who a message is directed toward.

    - is_direct_mention: bot username in text or reply to the bot.
    - reply_target_user_id set (and != speaker, != bot): reply to another user.
    - mentioned other users: addressed to them.
    - otherwise a general group message.
    """
    mentioned = set(mentioned_user_ids or [])
    text_lower = (text or "").lower()

    if is_direct_mention:
        return MessageTarget.DIRECTLY_TO_AIRI

    if reply_target_user_id is not None:
        if bot_user_id is not None and reply_target_user_id == bot_user_id:
            return MessageTarget.DIRECTLY_TO_AIRI
        if speaker_user_id is not None and reply_target_user_id != speaker_user_id:
            return MessageTarget.REPLY_TO_ANOTHER_USER
        return MessageTarget.UNCLEAR

    if mentioned:
        return MessageTarget.TO_ANOTHER_USER

    # "hey bob", "wait airi" style addressals without @ handles
    if bot_user_id is not None:
        pass  # username-based mentions are already captured via is_direct_mention

    return MessageTarget.GENERAL_GROUP_MESSAGE