from pathlib import Path
import os

from citation_deep_research.env import load_env_file


def test_load_env_file_does_not_override_existing_env(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SEMANTIC_SCHOLAR_API_KEY=file-key\nOPENALEX_MAILTO='me@example.edu'\n", encoding="utf-8")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "exported-key")
    monkeypatch.delenv("OPENALEX_MAILTO", raising=False)

    load_env_file(env_file)

    assert os.environ["SEMANTIC_SCHOLAR_API_KEY"] == "exported-key"
    assert os.environ["OPENALEX_MAILTO"] == "me@example.edu"
