"""Errors raised by the tools. Their message is written for the LLM / user to read and act on."""


class ToolError(ValueError):
    """A tool could not do what was asked (bad id, unsafe SQL, model missing, ...)."""
