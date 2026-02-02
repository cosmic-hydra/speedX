"""Tests package initialization."""
from speedx import RABoostClassifier, MASTClient, __version__


def test_imports():
    """Test that main classes can be imported."""
    assert RABoostClassifier is not None
    assert MASTClient is not None


def test_version():
    """Test version string."""
    assert __version__ == "0.1.0"
