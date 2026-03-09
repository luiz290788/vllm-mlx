# SPDX-License-Identifier: Apache-2.0
"""
Base engine interface for vllm-mlx inference.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Common HuggingFace config attribute names for context window size,
# ordered by prevalence across model architectures.
_CONTEXT_WINDOW_ATTRS = (
    "max_position_embeddings",  # LLaMA, Mistral, Qwen, GPT-NeoX, …
    "max_seq_len",  # Some older models
    "seq_length",  # GLM / ChatGLM
    "n_positions",  # GPT-2
    "max_sequence_length",  # Some custom configs
    "sliding_window",  # Mistral sliding window (fallback)
)


def extract_context_window(config: Any) -> int | None:
    """
    Extract the context window size from a HuggingFace model config.

    Tries common attribute names used by different model architectures.
    Returns ``None`` if the context window cannot be determined.
    """
    if config is None:
        return None

    # If config is a dict (e.g. from mlx-vlm load_config), use dict access
    if isinstance(config, dict):
        for attr in _CONTEXT_WINDOW_ATTRS:
            value = config.get(attr)
            if isinstance(value, int) and value > 0:
                return value
        # Check nested text_config for VLM models
        text_config = config.get("text_config", {})
        if isinstance(text_config, dict):
            for attr in _CONTEXT_WINDOW_ATTRS:
                value = text_config.get(attr)
                if isinstance(value, int) and value > 0:
                    return value
        return None

    # Object-style config (e.g. from mlx-lm)
    for attr in _CONTEXT_WINDOW_ATTRS:
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    # Check nested text_config for VLM models
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for attr in _CONTEXT_WINDOW_ATTRS:
            value = getattr(text_config, attr, None)
            if isinstance(value, int) and value > 0:
                return value

    return None


@dataclass
class GenerationOutput:
    """
    Output from generation.

    Compatible with both simple and batched engines.
    """

    text: str
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = "stop"
    # For streaming
    new_text: str = ""
    finished: bool = True
    # Per-token logprobs (mx.array of shape [vocab_size] for current token)
    logprobs: Any = None


class BaseEngine(ABC):
    """
    Abstract base class for inference engines.

    Both SimpleEngine and BatchedEngine implement this interface,
    allowing the server to use either without code changes.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @property
    @abstractmethod
    def is_mllm(self) -> bool:
        """Check if this is a multimodal model."""
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        pass

    @property
    def preserve_native_tool_format(self) -> bool:
        """
        Whether to preserve native tool message format.

        When True, role="tool" messages and tool_calls fields are preserved
        instead of being converted to text. Set by server based on tool parser.
        """
        return getattr(self, "_preserve_native_tool_format", False)

    @preserve_native_tool_format.setter
    def preserve_native_tool_format(self, value: bool) -> None:
        self._preserve_native_tool_format = value

    @abstractmethod
    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with complete text
        """
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Chat completion (non-streaming).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with assistant response
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream chat completion token by token.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        pass

    @property
    def context_window(self) -> int | None:
        """
        Get the model's context window size (max context length).

        Reads from the HuggingFace model config. Returns ``None`` if
        the context window cannot be determined.

        Subclasses may override this for custom logic.
        """
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics. Override in subclasses."""
        return {}

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics. Override in subclasses."""
        return None
