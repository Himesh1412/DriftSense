# tests/conftest.py
import pytest


@pytest.fixture(autouse=True)
def never_tag_the_real_repo(monkeypatch):
    """Tests that load the real config would otherwise create real Git tags (registry.git_tag_versions is on).
    versioning.py itself is still tested directly, with subprocess mocked."""
    monkeypatch.setattr("src.forecasting.train.tag_model_version", lambda *a, **k: False)
