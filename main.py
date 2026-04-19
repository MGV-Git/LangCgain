"""Демо-агент LangChain (погода + контекст пользователя + свой tool)."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any

# Консоль Windows (cp1251) ломает вывод UTF-8 без этой настройки
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde import _msgpack as _lg_msgpack
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Схема в отдельном модуле, чтобы в чекпоинте был стабильный путь `schemas.ResponseFormat`,
# а не `__main__.ResponseFormat` (он меняется при запуске файла как скрипт / как модуль).
from schemas import ResponseFormat

load_dotenv()

SYSTEM_PROMPT = """Ты РУССКОЯЗЫЧНЫЙ синоптик: весь ответ пользователю — одним связным русским текстом (поле text).

КРИТИЧЕСКОЕ ПРАВИЛО: только русский язык. Названия городов и стран — по-русски (например «Флорида», не Florida). Если пользователь пишет по-английски — всё равно отвечай по-русски.

Пиши естественно: например «Во Флориде сейчас ясно и солнечно…» — погода и место вплетены в обычные предложения, можно добавить лёгкий каламбур.

У тебя есть инструменты:

- get_weather_for_location: узнать погоду в указанном месте
- get_user_location: узнать местоположение пользователя по его ID (из контекста)
- get_travel_tip: короткий туристический совет «что взять с собой» для региона/города (демо)

Если пользователь спрашивает о погоде, сначала выясни место. Если явно «здесь» / «у меня» — вызови get_user_location.
Если спрашивают про поездку, что взять, совет туристу — можно вызвать get_travel_tip для города или региона."""


@tool
def get_weather_for_location(city: str) -> str:
    """Вернуть краткое демо-описание погоды в городе (для модели на русском)."""
    return f"В {city} по демо-данным сейчас ясно и солнечно."


@dataclass
class Context:
    """Custom runtime context schema."""

    user_id: str


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """По user_id вернуть регион пользователя (демо, для ответов на русском)."""
    user_id = runtime.context.user_id
    return "Флорида" if user_id == "1" else "Сан-Франциско"


@tool
def get_travel_tip(city_or_region: str) -> str:
    """Короткий демо-совет туристу: что учесть в поездке для названного города или региона."""
    key = city_or_region.strip().lower()
    hints: dict[str, str] = {
        "florida": "Жара и солнце: крем SPF, вода, лёгкая одежда; зонт от внезапного ливня не помешает.",
        "флорида": "Жара и солнце: крем SPF, вода, лёгкая одежда; зонт от внезапного ливня не помешает.",
        "sf": "Слои одежды: днём тепло, вечером прохладно и ветрено у залива.",
        "san francisco": "Слои одежды: днём тепло, вечером прохладно и ветрено у залива.",
        "сан-франциско": "Слои одежды: днём тепло, вечером прохладно и ветрено у залива.",
    }
    return hints.get(key, f"Для «{city_or_region}» (демо): уточните климат и сезон; возьмите удобную обувь и воду.")


def build_agent():
    model = init_chat_model(
        "openai:gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        max_tokens=1500,
        temperature=0.3,
    )

    # Явный allowlist убирает предупреждение LangGraph про «unregistered type» при десериализации.
    _allow = set(_lg_msgpack.SAFE_MSGPACK_TYPES) | {("schemas", "ResponseFormat")}
    checkpointer = InMemorySaver(
        serde=JsonPlusSerializer(allowed_msgpack_modules=tuple(_allow)),
    )

    return create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[get_user_location, get_weather_for_location, get_travel_tip],
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer,
    )


def print_answer(response: dict[str, Any]) -> None:
    """Печатает только текст ответа, без имён полей и repr модели."""
    sr = response["structured_response"]
    print(sr.text)


def run_demo(agent) -> None:
    config: dict[str, Any] = {"configurable": {"thread_id": "demo-1"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Какая сейчас погода на улице?"}]},
        config=config,
        context=Context(user_id="1"),
    )
    print_answer(response)

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Спасибо!"}]},
        config=config,
        context=Context(user_id="1"),
    )
    print_answer(response)


def run_chat(agent) -> None:
    """Интерактив: вводите сообщения; выход — пустая строка или Ctrl+Z / Ctrl+C."""
    config: dict[str, Any] = {"configurable": {"thread_id": "interactive-1"}}
    ctx = Context(user_id=os.getenv("DEMO_USER_ID", "1"))

    print("Чат с агентом. user_id из DEMO_USER_ID или по умолчанию 1. Пустая строка — выход.\n")
    while True:
        try:
            line = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        response = agent.invoke(
            {"messages": [{"role": "user", "content": line}]},
            config=config,
            context=ctx,
        )
        print_answer(response)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain demo agent (weather + custom tool)")
    parser.add_argument(
        "mode",
        nargs="?",
        default="demo",
        choices=("demo", "chat"),
        help="demo — два запроса как в уроке; chat — диалог в консоли",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Ошибка: не задан OPENAI_API_KEY. Создайте файл .env в папке проекта "
            "или задайте переменную окружения.",
            file=sys.stderr,
        )
        sys.exit(1)

    agent = build_agent()
    if args.mode == "chat":
        run_chat(agent)
    else:
        run_demo(agent)


if __name__ == "__main__":
    main()
