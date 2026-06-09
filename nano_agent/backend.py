"""Inference backend wrapping llama.cpp via llama-cpp-python.

The backend is the only component that touches the model. Keeping it isolated
means the agent loop can be unit-tested with a fake backend, and alternative
backends (gguf-npu, remote endpoints) can be dropped in later.
"""

from __future__ import annotations

from typing import Protocol


class Backend(Protocol):
    """Minimal interface the agent loop depends on."""

    def generate(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        ...


class LlamaCppBackend:
    """llama.cpp backend for GGUF models.

    On Jetson, build llama-cpp-python with CUDA (CMAKE_ARGS="-DGGML_CUDA=on")
    and set n_gpu_layers=-1 to offload all layers to the GPU.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        temperature: float = 0.1,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        # Imported lazily so the package imports without the heavy native dep
        # present (useful for tooling, tests with a fake backend, and CI lint).
        from llama_cpp import Llama

        self.temperature = temperature
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=verbose,
        )

    def generate(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        out = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=self.temperature,
            stop=stop,
        )
        return out["choices"][0]["text"]
