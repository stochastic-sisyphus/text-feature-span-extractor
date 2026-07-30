"""Projection-boundary Pandera base.

Producer-boundary schemas (TokensDF in tokenize.py, etc.) keep strict
config -- they catch real production bugs at parse time. Projection-
boundary schemas inherit from LenientDataFrameModel: coerce=True,
strict=False, library-native, never raises on drift.

Note: ``lazy=True`` is a ``.validate()`` call argument, not a Config
field. Pass it at the call site via the ``_lenient_validate`` helper in
``schemas.py``.
"""

from __future__ import annotations

import pandera.polars as ppa


class LenientDataFrameModel(ppa.DataFrameModel):
    """Projection-boundary DataFrameModel. Subclass instead of ppa.DataFrameModel
    for any schema that validates downstream of the producer seam."""

    class Config:
        strict = False
        coerce = True
