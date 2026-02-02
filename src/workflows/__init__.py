"""LlamaIndex Workflows integration."""

from src.workflows.document_workflow import (
    DocumentProcessingWorkflow,
    process_document_with_workflow,
)

__all__ = [
    "DocumentProcessingWorkflow",
    "process_document_with_workflow",
]
