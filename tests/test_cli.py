import pytest

from citation_deep_research.cli import main, parse_depth_caps


def test_cli_rejects_semantic_scholar_rate_above_one(tmp_path) -> None:
    code = main(["crawl", "--seed", "test", "--out", str(tmp_path), "--s2-rps", "2"])

    assert code == 2


def test_crawl_command_only_builds_configured_fallback_clients(monkeypatch, tmp_path) -> None:
    created = {"openalex": 0, "unpaywall": 0}

    class FakeSemanticScholarClient:
        def __init__(self, requests_per_second, max_retries, throttle_cooldown, retry_progress):
            self.requests_per_second = requests_per_second
            self.max_retries = max_retries
            self.throttle_cooldown = throttle_cooldown
            self.retry_progress = retry_progress

    class FakeOpenAlexClient:
        def __init__(self):
            created["openalex"] += 1

    class FakeUnpaywallClient:
        def __init__(self):
            created["unpaywall"] += 1

    class FakePipeline:
        def __init__(self, *, semantic_scholar, openalex, unpaywall):
            assert semantic_scholar.requests_per_second == 1.0
            assert semantic_scholar.max_retries == 8
            assert semantic_scholar.throttle_cooldown == 30.0
            assert callable(semantic_scholar.retry_progress)
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


def test_crawl_command_passes_semantic_scholar_throttle_options(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeSemanticScholarClient:
        def __init__(self, requests_per_second, max_retries, throttle_cooldown, retry_progress):
            captured["requests_per_second"] = requests_per_second
            captured["max_retries"] = max_retries
            captured["throttle_cooldown"] = throttle_cooldown
            captured["retry_progress"] = retry_progress

    class FakePipeline:
        def __init__(self, *, semantic_scholar, openalex, unpaywall):
            pass

        def crawl(self, **kwargs):
            captured.update(kwargs)

            class FakeManifest:
                def to_dict(self):
                    return {"counts": {}}

            return FakeManifest()

    monkeypatch.delenv("OPENALEX_MAILTO", raising=False)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    monkeypatch.setattr("citation_deep_research.cli.load_env_file", lambda: None)
    monkeypatch.setattr("citation_deep_research.cli.SemanticScholarClient", FakeSemanticScholarClient)
    monkeypatch.setattr("citation_deep_research.cli.CitationClosurePipeline", FakePipeline)

    code = main(
        [
            "crawl",
            "--seed",
            "test",
            "--out",
            str(tmp_path),
            "--s2-rps",
            "0.5",
            "--s2-max-retries",
            "12",
            "--s2-429-cooldown",
            "45",
            "--max-papers-to-expand",
            "30",
            "--quiet",
        ]
    )

    assert code == 0
    assert captured["requests_per_second"] == 0.5
    assert captured["max_retries"] == 12
    assert captured["throttle_cooldown"] == 45.0
    assert captured["retry_progress"] is None
    assert captured["semantic_scholar_requests_per_second"] == 0.5
    assert captured["semantic_scholar_max_retries"] == 12
    assert captured["semantic_scholar_429_cooldown"] == 45.0
    assert captured["max_papers_to_expand"] == 30
    assert captured["progress"] is None


def test_crawl_command_progress_enabled_by_default(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeSemanticScholarClient:
        def __init__(self, requests_per_second, max_retries, throttle_cooldown, retry_progress):
            captured["retry_progress"] = retry_progress

    class FakePipeline:
        def __init__(self, *, semantic_scholar, openalex, unpaywall):
            pass

        def crawl(self, **kwargs):
            captured.update(kwargs)

            class FakeManifest:
                def to_dict(self):
                    return {"counts": {}}

            return FakeManifest()

    monkeypatch.delenv("OPENALEX_MAILTO", raising=False)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key")
    monkeypatch.setattr("citation_deep_research.cli.load_env_file", lambda: None)
    monkeypatch.setattr("citation_deep_research.cli.SemanticScholarClient", FakeSemanticScholarClient)
    monkeypatch.setattr("citation_deep_research.cli.CitationClosurePipeline", FakePipeline)

    code = main(["crawl", "--seed", "test", "--out", str(tmp_path)])

    assert code == 0
    assert callable(captured["retry_progress"])
    assert callable(captured["progress"])


def test_crawl_command_reports_semantic_scholar_auth_status(monkeypatch, tmp_path, capsys) -> None:
    class FakeSemanticScholarClient:
        def __init__(self, requests_per_second, max_retries, throttle_cooldown, retry_progress):
            pass

    class FakePipeline:
        def __init__(self, *, semantic_scholar, openalex, unpaywall):
            pass

        def crawl(self, **kwargs):
            class FakeManifest:
                def to_dict(self):
                    return {"counts": {}}

            return FakeManifest()

    monkeypatch.delenv("OPENALEX_MAILTO", raising=False)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key")
    monkeypatch.setattr("citation_deep_research.cli.load_env_file", lambda: None)
    monkeypatch.setattr("citation_deep_research.cli.SemanticScholarClient", FakeSemanticScholarClient)
    monkeypatch.setattr("citation_deep_research.cli.CitationClosurePipeline", FakePipeline)

    code = main(["crawl", "--seed", "test", "--out", str(tmp_path)])

    assert code == 0
    assert "Semantic Scholar API key loaded: yes" in capsys.readouterr().err


def test_crawl_command_reports_runtime_failure_without_traceback(monkeypatch, tmp_path, capsys) -> None:
    class FakeSemanticScholarClient:
        def __init__(self, requests_per_second, max_retries, throttle_cooldown, retry_progress):
            pass

    class FakePipeline:
        def __init__(self, *, semantic_scholar, openalex, unpaywall):
            pass

        def crawl(self, **kwargs):
            raise RuntimeError("429 Client Error")

    monkeypatch.delenv("OPENALEX_MAILTO", raising=False)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    monkeypatch.setattr("citation_deep_research.cli.load_env_file", lambda: None)
    monkeypatch.setattr("citation_deep_research.cli.SemanticScholarClient", FakeSemanticScholarClient)
    monkeypatch.setattr("citation_deep_research.cli.CitationClosurePipeline", FakePipeline)

    code = main(["crawl", "--seed", "test", "--out", str(tmp_path)])

    assert code == 1
    assert "Crawl failed before outputs could be written: 429 Client Error" in capsys.readouterr().err


def test_cli_rejects_invalid_semantic_scholar_retry_options(tmp_path) -> None:
    assert main(["crawl", "--seed", "test", "--out", str(tmp_path), "--s2-rps", "0"]) == 2
    assert main(["crawl", "--seed", "test", "--out", str(tmp_path), "--s2-max-retries", "-1"]) == 2
    assert main(["crawl", "--seed", "test", "--out", str(tmp_path), "--s2-429-cooldown", "-1"]) == 2
    assert main(["crawl", "--seed", "test", "--out", str(tmp_path), "--max-papers-to-expand", "0"]) == 2


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
