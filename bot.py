import asyncio
import logging
import re
from typing import Optional

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart
from aiogram.types import Message

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_llm_client() -> Optional["OpenAI"]:
    if OpenAI and config.OPENROUTER_API_KEY:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,
        )
    return None


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

    # DM — everything is a query
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


async def ask_llm(client: "OpenAI", question: str) -> str:
    def _call() -> str:
        resp = client.chat.completions.create(
            model=config.OPENROUTER_MODEL,
            messages=[{"role": "user", "content": question}],
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

    @dp.message(CommandStart())
    async def on_start(message: Message) -> None:
        await message.answer(
            "Привет! Задай мне любой вопрос.\n"
            "В группе — тегни меня: @{} вопрос".format(me.username)
        )

    @dp.message(F.text)
    async def handle_query(message: Message) -> None:
        query = extract_query(message, bot_username)
        if not query:
            return

        placeholder = await message.answer("Думаю...")
        try:
            answer = await ask_llm(llm, query)
        except Exception as e:
            logger.exception("LLM call failed")
            await placeholder.edit_text(f"Ошибка: {type(e).__name__}: {e}")
            return

        await placeholder.delete()
        for chunk in split_message(answer):
            await message.answer(chunk, parse_mode=None)

    try:
        await dp.start_polling(bot)
    finally:
        pass


if __name__ == "__main__":
    asyncio.run(main())
