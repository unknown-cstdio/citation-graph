from __future__ import annotations

import re
import unicodedata
from collections import Counter
from math import sqrt

TOKEN_RE = re.compile(r"[a-z0-9]+")
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    normalized = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return " ".join(TOKEN_RE.findall(normalized.lower()))


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    value = doi.strip()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^doi:\s*", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".,;")
    return value.lower() or None


def extract_doi(value: str) -> str | None:
    match = DOI_RE.search(value)
    return normalize_doi(match.group(0)) if match else None


def token_counts(text: str | None) -> Counter[str]:
    return Counter(TOKEN_RE.findall((text or "").lower()))


def cosine_counts(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def cosine_vectors(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
