"""Errors shared by image I/O and the standalone worker."""


class CountingError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class CountingCancelled(Exception):
    """Cooperative cancellation before review publication."""
