"""Mock external dependencies, served over real HTTP: Disseqt and an OpenAI-compatible LLM."""

from .app import create_mock_app

__all__ = ["create_mock_app"]
