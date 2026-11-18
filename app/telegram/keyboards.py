from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def create_keyboards():
    return {
        "main_menu": InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Stats", callback_data="stats"),
             InlineKeyboardButton("🧠 Memories", callback_data="memories")],
            [InlineKeyboardButton("💭 Mood", callback_data="mood"),
             InlineKeyboardButton("👥 Relationships", callback_data="relationships")],
        ]),
    }