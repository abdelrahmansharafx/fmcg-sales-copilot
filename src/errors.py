"""Errors raised by the tools. Their message is written for the LLM / user to read and act on.

The subclasses let the API pick the right HTTP status (404 / 503) without parsing messages.
"""


class ToolError(ValueError):
    """A tool could not do what was asked (bad input, unsafe SQL, ...). HTTP 422."""


class NotFoundError(ToolError):
    """An id or combination does not exist (unknown outlet, SKU not stocked, ...). HTTP 404."""


class NotReadyError(ToolError):
    """The service is not set up for this (no model, no database, no LLM key). HTTP 503."""
