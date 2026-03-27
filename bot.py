import asyncio
import json
import logging
import os
import re
from collections import deque
from typing import Optional

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_HISTORY = 20  # messages per chat (10 pairs of Q&A)
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.json")


def load_histories() -> dict[str, list[dict]]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_histories(histories: dict[str, deque]) -> None:
    data = {k: list(v) for k, v in histories.items()}
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_llm_client() -> Optional["OpenAI"]:
    if OpenAI and config.OPENROUTER_API_KEY:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,
        )
    return None


def get_display_name(message: Message) -> str:
    user = message.from_user
    if not user:
        return "Unknown"
    if user.first_name and user.last_name:
        return f"{user.first_name} {user.last_name}"
    return user.first_name or user.username or "Unknown"


def extract_query(message: Message, bot_username: str) -> Optional[str]:
    text = (message.text or "").strip()
    if not text:
        return None

    if message.chat.type in {"group", "supergroup"}:
        entities = message.entities or []
        mentions = [
            text[e.offset : e.offset + e.length]
            for e in entities
            if e.type == "mention"
        ]
        has_mention = any(m.lower() == f"@{bot_username}" for m in mentions)
        pattern = re.compile(rf"@{re.escape(bot_username)}", re.IGNORECASE)
        if not (has_mention or pattern.search(text)):
            return None
        cleaned = pattern.sub("", text).strip()
        return cleaned or None

    return text


def split_message(text: str, limit: int = 4096) -> list[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    while len(text) > limit:
        split_at = text.rfind("\n\n", 0, limit)
        if split_at == -1:
            split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = text.rfind(" ", 0, limit)
        if split_at == -1:
            split_at = limit
        chunk = text[:split_at].strip()
        if not chunk:
            chunk = text[:limit]
            split_at = limit
        parts.append(chunk)
        text = text[split_at:].lstrip()
    if text:
        parts.append(text)
    return parts


async def ask_llm(
    client: "OpenAI",
    history: list[dict],
) -> str:
    def _call() -> str:
        resp = client.chat.completions.create(
            model=config.OPENROUTER_MODEL,
            messages=history,
        )
        return resp.choices[0].message.content.strip()

    return await asyncio.to_thread(_call)


async def main() -> None:
    if not config.BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is missing. Add it to .env")

    llm = build_llm_client()
    if not llm:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to .env")

    bot = Bot(
        token=config.BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
    )
    dp = Dispatcher()

    me = await bot.get_me()
    bot_username = (me.username or "").lower()
    logger.info(f"Bot started as @{bot_username}")

    # Load history from disk
    raw = load_histories()
    chat_histories: dict[str, deque] = {}
    for k, v in raw.items():
        d = deque(v, maxlen=MAX_HISTORY)
        chat_histories[k] = d

    @dp.message(CommandStart())
    async def on_start(message: Message) -> None:
        await message.answer(
            "Привет! Задай мне любой вопрос.\n"
            "В группе — тегни меня: @{} вопрос\n"
            "/clear — очистить историю диалога".format(me.username)
        )

    @dp.message(Command("clear"))
    async def on_clear(message: Message) -> None:
        key = str(message.chat.id)
        if key in chat_histories:
            chat_histories[key].clear()
        save_histories(chat_histories)
        await message.answer("История очищена.")

    @dp.message(F.text)
    async def handle_query(message: Message) -> None:
        query = extract_query(message, bot_username)
        if not query:
            return

        # In groups, prefix with user name so LLM knows who's asking
        is_group = message.chat.type in {"group", "supergroup"}
        user_name = get_display_name(message)
        user_content = f"[{user_name}]: {query}" if is_group else query

        key = str(message.chat.id)
        if key not in chat_histories:
            chat_histories[key] = deque(maxlen=MAX_HISTORY)
        history = chat_histories[key]
        history.append({"role": "user", "content": user_content})

        # Build messages list from history
        messages = list(history)

        placeholder = await message.answer("Думаю...")
        try:
            answer = await ask_llm(llm, messages)
        except Exception as e:
            logger.exception("LLM call failed")
            history.pop()  # remove failed question
            await placeholder.edit_text(f"Ошибка: {type(e).__name__}: {e}")
            return

        history.append({"role": "assistant", "content": answer})
        save_histories(chat_histories)

        await placeholder.delete()
        for chunk in split_message(answer):
            await message.answer(chunk, parse_mode=None)

    try:
        await dp.start_polling(bot)
    finally:
        pass


if __name__ == "__main__":
    asyncio.run(main())
