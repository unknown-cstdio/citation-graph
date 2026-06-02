from __future__ import annotations

from typing import Any

import requests


class OllamaEmbeddingClient:
    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def embed(self, text: str) -> list[float]:
        response = self.session.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        embeddings = data.get("embeddings")
        if embeddings and isinstance(embeddings, list):
            return list(embeddings[0])
        embedding = data.get("embedding")
        if embedding and isinstance(embedding, list):
            return list(embedding)
        raise RuntimeError(f"Ollama response did not include an embedding for model {self.model!r}")
