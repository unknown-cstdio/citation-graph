from citation_deep_research.ollama import OllamaEmbeddingClient


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {"embeddings": [[1.0, 2.0, 3.0]]}


class FakeSession:
    def __init__(self) -> None:
        self.request = None

    def post(self, url, json, timeout):
        self.request = {"url": url, "json": json, "timeout": timeout}
        return FakeResponse()


def test_ollama_embedding_client_uses_embed_endpoint() -> None:
    client = OllamaEmbeddingClient(model="nomic-embed-text", base_url="http://ollama.test", timeout=12)
    fake_session = FakeSession()
    client.session = fake_session

    embedding = client.embed("hello")

    assert embedding == [1.0, 2.0, 3.0]
    assert fake_session.request["url"] == "http://ollama.test/api/embed"
    assert fake_session.request["json"] == {"model": "nomic-embed-text", "input": "hello"}
