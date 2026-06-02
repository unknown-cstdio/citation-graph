import pytest

from citation_deep_research.cli import main, parse_depth_caps


def test_cli_rejects_semantic_scholar_rate_above_one(tmp_path) -> None:
    code = main(["crawl", "--seed", "test", "--out", str(tmp_path), "--s2-rps", "2"])

    assert code == 2


def test_crawl_command_only_builds_configured_fallback_clients(monkeypatch, tmp_path) -> None:
    created = {"openalex": 0, "unpaywall": 0}

    class FakeSemanticScholarClient:
        def __init__(self, requests_per_second):
            self.requests_per_second = requests_per_second

    class FakeOpenAlexClient:
        def __init__(self):
            created["openalex"] += 1

    class FakeUnpaywallClient:
        def __init__(self):
            created["unpaywall"] += 1

    class FakePipeline:
        def __init__(self, *, semantic_scholar, openalex, unpaywall):
            assert semantic_scholar.requests_per_second == 1.0
            assert openalex is None
            assert unpaywall is None

        def crawl(self, **kwargs):
            class FakeManifest:
                def to_dict(self):
                    return {"counts": {}}

            return FakeManifest()

    monkeypatch.delenv("OPENALEX_MAILTO", raising=False)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    monkeypatch.setattr("citation_deep_research.cli.load_env_file", lambda: None)
    monkeypatch.setattr("citation_deep_research.cli.SemanticScholarClient", FakeSemanticScholarClient)
    monkeypatch.setattr("citation_deep_research.cli.OpenAlexClient", FakeOpenAlexClient)
    monkeypatch.setattr("citation_deep_research.cli.UnpaywallClient", FakeUnpaywallClient)
    monkeypatch.setattr("citation_deep_research.cli.CitationClosurePipeline", FakePipeline)

    code = main(["crawl", "--seed", "test", "--out", str(tmp_path)])

    assert code == 0
    assert created == {"openalex": 0, "unpaywall": 0}


def test_parse_depth_caps() -> None:
    assert parse_depth_caps("40,30") == [40, 30]
    assert parse_depth_caps(None) is None


def test_cli_rejects_removed_expand_selected_only_flag(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["crawl", "--seed", "test", "--out", str(tmp_path), "--expand-selected-only"])

    assert exc_info.value.code == 2


def test_cli_rejects_removed_ask_command() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["ask", "--run", "runs/example", "--question", "test"])

    assert exc_info.value.code == 2
