from citation_deep_research.clients import relation_warning


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
