from citation_deep_research.clients import HttpClient, relation_warning


def test_relation_warning_detects_publisher_elision() -> None:
    warning = relation_warning(
        "paper-1",
        "references",
        {
            "data": None,
            "citingPaperInfo": {
                "title": "Seed",
                "openAccessPdf": {
                    "disclaimer": "Notice: The following paper fields have been elided by the publisher: {'references'}."
                },
            },
        },
    )

    assert warning is not None
    assert warning.code == "references_publisher_elided"
    assert warning.paper_id == "paper-1"
    assert warning.relation == "references"


def test_http_client_emits_retry_progress() -> None:
    messages = []
    client = HttpClient(
        base_url="https://example.test",
        max_retries=8,
        retry_progress=messages.append,
    )

    client.emit_retry_progress("https://example.test/paper", 429, 0, 30.0)

    assert messages == ["Semantic Scholar request got 429; retry 1/8 in 30.0s: https://example.test/paper"]
