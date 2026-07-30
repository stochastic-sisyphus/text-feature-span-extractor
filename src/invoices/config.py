"""Configuration bridge — re-exports Settings as Config for compatibility.

Usage unchanged:
    from invoices.config import Config

    threshold = Config.confidence_auto_approve
    none_bias = Config.none_bias
"""

from .settings import Settings
from .settings import settings as Config

__all__ = ["Config", "Settings"]
