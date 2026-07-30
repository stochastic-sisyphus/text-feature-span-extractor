"""emit package — contract emission and review-queue construction."""

from ._constants import EVALUATOR_VERSION
from ._document import emit_document_with_data
from ._field_output import compute_field_confidence, create_field_output
from ._priority import compute_priority_score, create_evaluation_entry

__all__ = [
    "EVALUATOR_VERSION",
    "compute_field_confidence",
    "compute_priority_score",
    "create_evaluation_entry",
    "create_field_output",
    "emit_document_with_data",
]
