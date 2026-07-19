"""Application-specific errors that can be shown directly to CLI users."""


class NekoTaggerError(Exception):
    """Base class for expected application failures."""


class InputError(NekoTaggerError):
    """The input path or option combination is invalid."""


class ConflictError(NekoTaggerError):
    """Output conflicts could not be resolved safely."""


class ModelError(NekoTaggerError):
    """A model could not be downloaded, loaded, or executed."""


class MetadataError(NekoTaggerError):
    """Tag metadata could not be downloaded, loaded, or validated."""
