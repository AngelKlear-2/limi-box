"""Anthropic Messages API backend."""

from __future__ import annotations

from limi.llm.base import LLMProvider, ProviderUnavailable


class AnthropicLLM(LLMProvider):
    """Needs config: llm.anthropic.model and llm.anthropic.api_key (or env ANTHROPIC_API_KEY)."""

    def __init__(self, model: str, api_key: str, temperature: float, max_tokens: int) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderUnavailable("The 'anthropic' package is missing. Run: pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=api_key, timeout=60.0, max_retries=1)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _complete(self, messages: list[dict[str, str]]) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] in ("user", "assistant")]
        while convo and convo[0]["role"] != "user":  # the API wants a user turn first
            convo.pop(0)
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system,
            messages=convo,  # type: ignore[arg-type]
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
