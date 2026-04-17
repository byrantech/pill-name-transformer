from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import requests
from wordfreq import zipf_frequency

OPENFDA_ENDPOINT = "https://api.fda.gov/drug/label.json"
DEFAULT_CURATED_INCLUDE_PATH = Path("data/curated_brand_names.txt")
DEFAULT_CURATED_BLOCKLIST_PATH = Path("data/brand_blocklist.txt")
BLOCKED_TOKENS = {
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "injection",
    "solution",
    "syrup",
    "cream",
    "gel",
    "lotion",
    "spray",
    "drops",
    "patch",
    "kit",
    "pack",
    "extra",
    "strength",
    "plus",
    "cold",
    "flu",
    "pain",
    "relief",
    "adult",
    "childrens",
    "children",
    "maximum",
    "pm",
    "am",
    "xr",
    "er",
    "sr",
    "cr",
    "dr",
    "hcl",
}


def _normalize_name(raw: str) -> str | None:
    name = raw.strip().lower()
    name = re.sub(r"[^a-z0-9\- ]+", "", name)
    name = re.sub(r"\s+", " ", name)
    if len(name) < 3:
        return None
    if any(ch.isdigit() for ch in name):
        return None
    return name


def _load_word_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    words: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        normalized = _normalize_name(cleaned)
        if normalized:
            words.add(normalized)
    return words


def _is_brand_like_name(name: str, generic_terms: set[str]) -> bool:
    tokens = name.split()
    # Keep this very strict: invented brand names are usually single tokens.
    if len(tokens) != 1:
        return False
    for token in tokens:
        if not token.isalpha():
            return False
        if not (5 <= len(token) <= 12):
            return False
        if token in BLOCKED_TOKENS:
            return False
        if token in generic_terms:
            return False
        # Keep only tokens that are essentially "not English words".
        # This aggressively favors invented brand-like names.
        if zipf_frequency(token, "en") > 0.0:
            return False
    if name in generic_terms:
        return False
    return True


def fetch_openfda_names(
    max_records: int = 5000,
    page_size: int = 100,
    curated_include_path: Path = DEFAULT_CURATED_INCLUDE_PATH,
    curated_blocklist_path: Path = DEFAULT_CURATED_BLOCKLIST_PATH,
) -> list[str]:
    names: set[str] = set()
    generic_terms: set[str] = set()
    curated_include = _load_word_list(curated_include_path)
    curated_blocklist = _load_word_list(curated_blocklist_path)
    skip = 0
    while skip < max_records:
        response = requests.get(
            OPENFDA_ENDPOINT,
            params={"limit": page_size, "skip": skip},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        if not results:
            break
        for item in results:
            ofda = item.get("openfda", {})
            for key in ("generic_name", "substance_name"):
                for candidate in ofda.get(key, []):
                    normalized = _normalize_name(candidate)
                    if normalized:
                        generic_terms.add(normalized)
                        generic_terms.update(normalized.split())
            for candidate in ofda.get("brand_name", []):
                normalized = _normalize_name(candidate)
                if normalized and _is_brand_like_name(normalized, generic_terms):
                    names.add(normalized)
        skip += page_size
    names.update(curated_include)
    names = {name for name in names if name not in curated_blocklist}
    return sorted(names)


def save_names(names: Iterable[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = sorted(set(names))
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_names(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))
