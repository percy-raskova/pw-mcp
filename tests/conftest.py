"""Shared pytest fixtures for pw-mcp tests."""

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MEDIAWIKI_DIR = FIXTURES_DIR / "mediawiki"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def mediawiki_dir() -> Path:
    """Return path to MediaWiki fixtures."""
    return MEDIAWIKI_DIR


@pytest.fixture
def load_fixture() -> callable:
    """Factory fixture to load MediaWiki fixture files.

    Usage:
        def test_something(load_fixture):
            content = load_fixture("infoboxes", "politician.txt")
    """

    def _load(category: str, name: str) -> str:
        path = MEDIAWIKI_DIR / category / name
        return path.read_text(encoding="utf-8")

    return _load


@pytest.fixture
def politician_infobox(load_fixture: callable) -> str:
    """Sample politician infobox (Abraham Lincoln)."""
    return load_fixture("infoboxes", "politician.txt")


@pytest.fixture
def country_infobox(load_fixture: callable) -> str:
    """Sample country infobox (Kingdom of Norway)."""
    return load_fixture("infoboxes", "country.txt")


@pytest.fixture
def political_party_infobox(load_fixture: callable) -> str:
    """Sample political party infobox (Communist Party of Germany)."""
    return load_fixture("infoboxes", "political_party.txt")


@pytest.fixture
def person_infobox(load_fixture: callable) -> str:
    """Sample person infobox (Jean-Francois Champollion)."""
    return load_fixture("infoboxes", "person.txt")


@pytest.fixture
def philosopher_infobox(load_fixture: callable) -> str:
    """Sample philosopher infobox (Ludwig Feuerbach)."""
    return load_fixture("infoboxes", "philosopher.txt")


@pytest.fixture
def organization_infobox(load_fixture: callable) -> str:
    """Sample organization infobox (Central Military Commission)."""
    return load_fixture("infoboxes", "organization.txt")


@pytest.fixture
def youtuber_infobox(load_fixture: callable) -> str:
    """Sample youtuber infobox (The Kavernacle)."""
    return load_fixture("infoboxes", "youtuber.txt")


@pytest.fixture
def web_citation(load_fixture: callable) -> str:
    """Sample web citation template."""
    return load_fixture("citations", "web_citation.txt")


@pytest.fixture
def citation_template(load_fixture: callable) -> str:
    """Sample citation template."""
    return load_fixture("citations", "citation_template.txt")


@pytest.fixture
def simple_links(load_fixture: callable) -> str:
    """Sample simple internal links."""
    return load_fixture("links", "simple_links.txt")


@pytest.fixture
def piped_links(load_fixture: callable) -> str:
    """Sample piped internal links [[Target|Display]]."""
    return load_fixture("links", "piped_links.txt")


@pytest.fixture
def category_links(load_fixture: callable) -> str:
    """Sample category links [[Category:Name]]."""
    return load_fixture("links", "category_links.txt")


@pytest.fixture
def unicode_infobox(load_fixture: callable) -> str:
    """Sample infobox with Unicode characters (Suppiluliuma I)."""
    return load_fixture("edge_cases", "unicode_infobox.txt")


# =============================================================================
# SEMBR FIXTURES
# =============================================================================

SEMBR_FIXTURES_DIR = FIXTURES_DIR / "sembr"


@pytest.fixture
def sembr_fixtures_dir() -> Path:
    """Return path to sembr fixtures directory."""
    return SEMBR_FIXTURES_DIR


@pytest.fixture
def sembr_input_dir() -> Path:
    """Return path to sembr input fixtures."""
    return SEMBR_FIXTURES_DIR / "input"


@pytest.fixture
def sembr_expected_dir() -> Path:
    """Return path to sembr expected output fixtures."""
    return SEMBR_FIXTURES_DIR / "expected"


@pytest.fixture
def sembr_mock_responses_dir() -> Path:
    """Return path to sembr mock HTTP responses."""
    return SEMBR_FIXTURES_DIR / "mock_responses"


@pytest.fixture
def load_sembr_fixture() -> callable:
    """Factory fixture to load sembr fixture files.

    Usage:
        def test_something(load_sembr_fixture):
            content = load_sembr_fixture("input", "simple_english.txt")
    """

    def _load(category: str, name: str) -> str:
        path = SEMBR_FIXTURES_DIR / category / name
        return path.read_text(encoding="utf-8")

    return _load


@pytest.fixture
def load_sembr_mock_response() -> callable:
    """Factory fixture to load sembr mock HTTP responses.

    Usage:
        def test_something(load_sembr_mock_response):
            response = load_sembr_mock_response("success.json")
    """

    def _load(name: str) -> dict[str, Any]:
        path = SEMBR_FIXTURES_DIR / "mock_responses" / name
        return json.loads(path.read_text(encoding="utf-8"))

    return _load


@pytest.fixture
def mock_health_response(load_sembr_mock_response: callable) -> dict[str, str]:
    """Pre-loaded health check mock response."""
    return load_sembr_mock_response("health_check.json")


@pytest.fixture
def mock_rewrap_response(load_sembr_mock_response: callable) -> dict[str, str]:
    """Pre-loaded rewrap success mock response."""
    return load_sembr_mock_response("success.json")


@pytest.fixture
def mock_error_response(load_sembr_mock_response: callable) -> dict[str, str]:
    """Pre-loaded error mock response."""
    return load_sembr_mock_response("error.json")
