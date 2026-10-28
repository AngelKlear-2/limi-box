"""Local GGUF model through llama-cpp-python."""

from __future__ import annotations

from pathlib import Path

from limi.llm.base import LLMProvider, ProviderUnavailable


class LocalGGUFLLM(LLMProvider):
    """Needs config: llm.local_gguf.model_path, n_ctx, n_gpu_layers, plus llm.temperature.

    The .gguf file is placed in models/ by the user; Limi never downloads it.
    """

    def __init__(
        self, model_path: Path, n_ctx: int, n_gpu_layers: int, temperature: float, max_tokens: int
    ) -> None:
        if not model_path.is_file():
            raise ProviderUnavailable(
                f"GGUF model not found. Expected a file at:\n  {model_path}\n"
                "Put a .gguf model there, or set llm.provider: api in config.yaml."
            )
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ProviderUnavailable(
                "llama-cpp-python is not installed (it is an optional extra).\n"
                "Run: pip install llama-cpp-python   (see README for GPU and Raspberry Pi notes)"
            ) from exc
        self._llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _complete(self, messages: list[dict[str, str]]) -> str:
        out = self._llm.create_chat_completion(
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return out["choices"][0]["message"]["content"] or ""  # type: ignore[index]
