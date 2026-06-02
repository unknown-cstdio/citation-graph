from citation_deep_research.clients import UnpaywallClient


def test_unpaywall_lookup_treats_404_as_missing_record() -> None:
    class FakeHttp:
        def get(self, path, *, params):
            raise RuntimeError("404 Client Error: Not Found")

    client = UnpaywallClient(email="me@example.edu")
    client.http = FakeHttp()

    assert client.lookup("10.1234/missing") is None
