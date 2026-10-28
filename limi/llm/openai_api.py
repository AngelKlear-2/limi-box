"""OpenAI-compatible chat backend (OpenAI, DeepSeek, OpenRouter, Ollama, LM Studio...)."""

from __future__ import annotations

from limi.llm.base import LLMProvider, ProviderUnavailable


class OpenAICompatLLM(LLMProvider):
    """Needs config: llm.api.base_url, llm.api.model, llm.api.api_key.

    The key can also come from env LIMI_API_KEY (env wins). Local servers
    such as Ollama or LM Studio accept any key, so an empty one is fine there.
    """

    def __init__(self, base_url: str, model: str, api_key: str, temperature: float) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderUnavailable("The 'openai' package is missing. Run: pip install openai") from exc
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=60.0, max_retries=1)
        self._model = model
        self._temperature = temperature

    def _complete(self, messages: list[dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
        )
        return resp.choices[0].message.content or ""
