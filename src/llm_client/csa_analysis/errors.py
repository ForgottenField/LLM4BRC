"""Custom exceptions for CSA analysis."""


class CSAError(Exception):
    """Base exception for all CSA analysis errors."""


class CSANotFoundError(CSAError):
    """CSA (clang/scan-build) is not installed or not in PATH."""


class SourceFileNotFoundError(CSAError):
    """The target source file does not exist."""


class CSATimeoutError(CSAError):
    """CSA analysis exceeded the configured timeout."""


class PlistParseError(CSAError):
    """Failed to parse CSA plist output."""


class CallGraphBuildError(CSAError):
    """Failed to build or load the project call graph."""
