import os
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ChatModelSpec:
    name: str
    env_key: str
    default_model: str
    display_name: str = ""

    @property
    def label(self) -> str:
        return self.display_name or self.name.title()


CHAT_MODELS: tuple[ChatModelSpec, ...] = (
    ChatModelSpec(
        name="dashscope",
        env_key="DASHSCOPE_API_KEY",
        default_model="qwen-plus",
        display_name="DashScope (通义千问)",
    ),
    ChatModelSpec(
        name="openai",
        env_key="OPENAI_API_KEY",
        default_model="gpt-4o",
        display_name="OpenAI",
    ),
    ChatModelSpec(
        name="zhipuai",
        env_key="ZHIPUAI_API_KEY",
        default_model="glm-4-flash",
        display_name="智谱AI",
    ),
    ChatModelSpec(
        name="deepseek",
        env_key="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        display_name="DeepSeek",
    ),
)

_DEFAULT_PROVIDER = "dashscope"

_registry: dict[str, ChatModelSpec] = {}
_builders: dict[str, Callable[..., Any]] = {}


def _build_registry() -> dict[str, ChatModelSpec]:
    if not _registry:
        for spec in CHAT_MODELS:
            _registry[spec.name] = spec
    return _registry


def find_by_name(name: str) -> ChatModelSpec:
    reg = _build_registry()
    spec = reg.get(name)
    if spec is None:
        available = ", ".join(reg.keys())
        raise ValueError(f"Unknown chat model provider: '{name}'. Available: {available}")
    return spec


def register_builder(name: str):
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _builders[name] = fn
        return fn
    return decorator


def create_chat_model(
    provider: str | None = None,
    model: str | None = None,
):
    provider = provider or os.getenv("CHAT_MODEL_PROVIDER", _DEFAULT_PROVIDER)
    spec = find_by_name(provider)
    api_key = os.getenv(spec.env_key)
    model = model or os.getenv("CHAT_MODEL_NAME", spec.default_model)

    builder = _builders.get(provider)
    if builder is None:
        raise ValueError(f"No builder registered for provider: '{provider}'")

    return builder(model=model, api_key=api_key)


@register_builder("dashscope")
def _build_dashscope(model: str, api_key: str, **kwargs):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


@register_builder("openai")
def _build_openai(model: str, api_key: str, **kwargs):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, api_key=api_key)


@register_builder("zhipuai")
def _build_zhipuai(model: str, api_key: str, **kwargs):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )


@register_builder("deepseek")
def _build_deepseek(model: str, api_key: str, **kwargs):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


chat_model = create_chat_model()
