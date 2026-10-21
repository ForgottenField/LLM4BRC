"""Static knowledge base for standard library / POSIX function summaries.

When the CFG analysis encounters a function call to a standard library
function (no body available in the project source), it looks up the
function here first rather than relying on an LLM best-guess summary.

Adding entries:
    ``_LIB_FUNCTIONS[function_name] = LibraryFuncSummary(...)``

For functions not yet in the knowledge base, the analysis will raise a
clear error directing the developer to add documentation or the function
body.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LibraryFuncSummary(BaseModel):
    """Summary of a standard library function's behavior relevant to
    path feasibility analysis."""

    function_name: str = Field(description="Function name")
    header: str = Field(description="C/C++ header to include", default="")
    signature: str = Field(description="Function signature")
    return_value_semantics: str = Field(
        description="What different return values mean"
    )
    side_effects: str = Field(
        description="Side effects: writes through pointer params, initializes output, etc."
    )
    writes_through_params: list[str] = Field(
        default_factory=list,
        description="Parameter names that may be written to (output params)",
    )
    preconditions: list[str] = Field(
        default_factory=list,
        description="Preconditions that must hold",
    )
    analysis: str = Field(
        description="Free-text analysis relevant to path feasibility"
    )
    source: str = Field(
        default="POSIX standard",
        description="Source of this information",
    )

    def format_for_llm(self) -> str:
        """Format this summary for inclusion in the LLM precondition prompt."""
        lines = [
            f"### {self.function_name}",
            f"* Header: `{self.header}`",
            f"* Signature: `{self.signature}`",
            f"* Return semantics: {self.return_value_semantics}",
        ]
        if self.side_effects:
            lines.append(f"* Side effects: {self.side_effects}")
        if self.writes_through_params:
            for p in self.writes_through_params:
                lines.append(f"  - writes through: `{p}`")
        if self.preconditions:
            lines.append("* Preconditions:")
            for p in self.preconditions:
                lines.append(f"  - {p}")
        if self.analysis:
            lines.append(f"* Analysis: {self.analysis}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Knowledge base: function_name → LibraryFuncSummary
# ---------------------------------------------------------------------------

_LIB_FUNCTIONS: dict[str, LibraryFuncSummary] = {}


def register(summary: LibraryFuncSummary) -> None:
    """Register a library function summary."""
    _LIB_FUNCTIONS[summary.function_name] = summary


def lookup(function_name: str) -> LibraryFuncSummary | None:
    """Look up a standard library function summary by name."""
    return _LIB_FUNCTIONS.get(function_name)


def list_registered() -> list[str]:
    """Return sorted list of registered function names."""
    return sorted(_LIB_FUNCTIONS.keys())


# ===================================================================
# POSIX / Standard C Library functions
# ===================================================================

register(LibraryFuncSummary(
    function_name="getopt_long",
    header="<getopt.h>",
    signature="int getopt_long(int argc, char *const argv[], const char *optstring, const struct option *longopts, int *longindex);",
    return_value_semantics=(
        "Returns the short option character on matching a short option. "
        "Returns 0 when a long option with ``flag != NULL`` is matched — "
        "in this case ``*longopts[i].flag`` is set to ``longopts[i].val``. "
        "Returns -1 when all options have been processed (end of options). "
        "Returns '?' (63) on an unknown option or missing argument."
    ),
    side_effects=(
        "When a long option with ``flag != NULL`` is matched, writes ``val`` "
        "to the location pointed to by ``flag`` (i.e. ``*opt->flag = opt->val``). "
        "``optind``, ``optopt``, and ``opterr`` globals may be updated. "
        "``argv`` and ``argc`` are read, not modified."
    ),
    writes_through_params=["longopts[i].flag (indirect write)"],
    preconditions=[
        "longopts must be an array of struct option, terminated by an entry with name=NULL",
        "optstring must be a valid option string",
    ],
    analysis=(
        "CRITICAL: getopt_long returns 0 ONLY when a long option with a non-null flag "
        "pointer is matched. In this case, it writes to ``*flag``. This write initializes "
        "the variable that ``flag`` points to. If ``longopts`` entries have ``flag = NULL`` "
        "(zero-initialized), then getopt_long CANNOT return 0 for those entries — they "
        "will return the option's val instead. The only way to get return value 0 is when "
        "an option with ``flag != NULL`` exists in the array AND is matched."
    ),
))

register(LibraryFuncSummary(
    function_name="getopt",
    header="<unistd.h>",
    signature="int getopt(int argc, char *const argv[], const char *optstring);",
    return_value_semantics=(
        "Returns the next option character from argv. "
        "Returns -1 when all options have been processed. "
        "Returns '?' on an unknown option."
    ),
    side_effects="Updates optind, optopt, and opterr globals.",
    writes_through_params=[],
    preconditions=[],
    analysis="getopt is the short-option-only version. It never returns 0; the return value is always the option character, -1, or '?'.",
))

register(LibraryFuncSummary(
    function_name="memcpy",
    header="<cstring> / <string.h>",
    signature="void *memcpy(void *dest, const void *src, size_t n);",
    return_value_semantics="Returns dest (the destination pointer).",
    side_effects="Writes n bytes to the memory pointed to by dest, reading from src.",
    writes_through_params=["dest"],
    preconditions=["dest must be writable for n bytes", "src must be readable for n bytes", "dest and src must not overlap"],
    analysis="memcpy unconditionally writes n bytes to dest. After the call, the memory at dest is fully initialized.",
))

register(LibraryFuncSummary(
    function_name="memset",
    header="<cstring> / <string.h>",
    signature="void *memset(void *s, int c, size_t n);",
    return_value_semantics="Returns s (the destination pointer).",
    side_effects="Writes c (converted to unsigned char) to the first n bytes of s.",
    writes_through_params=["s"],
    preconditions=["s must be writable for n bytes"],
    analysis="memset unconditionally writes n bytes to s. After the call, the memory at s is initialized.",
))

register(LibraryFuncSummary(
    function_name="malloc",
    header="<cstdlib> / <stdlib.h>",
    signature="void *malloc(size_t size);",
    return_value_semantics="Returns a pointer to the allocated memory on success, NULL on failure (out of memory).",
    side_effects="Allocates size bytes of uninitialized memory from the heap. The memory contents are indeterminate.",
    writes_through_params=[],
    preconditions=["size may be 0 (implementation-defined behavior)"],
    analysis="malloc allocates memory but does NOT initialize it. The returned memory contains indeterminate values. The caller must initialize it before reading.",
))

register(LibraryFuncSummary(
    function_name="calloc",
    header="<cstdlib> / <stdlib.h>",
    signature="void *calloc(size_t nmemb, size_t size);",
    return_value_semantics="Returns a pointer to the allocated memory on success, NULL on failure.",
    side_effects="Allocates nmemb*size bytes of zero-initialized memory from the heap.",
    writes_through_params=[],
    preconditions=[],
    analysis="Unlike malloc, calloc zero-initializes the allocated memory. The returned memory is safe to read without further initialization.",
))

register(LibraryFuncSummary(
    function_name="realloc",
    header="<cstdlib> / <stdlib.h>",
    signature="void *realloc(void *ptr, size_t size);",
    return_value_semantics="Returns a pointer to the resized memory on success, NULL on failure (original ptr unchanged).",
    side_effects="Resizes the memory block pointed to by ptr. New bytes (if growing) are uninitialized.",
    writes_through_params=[],
    preconditions=["ptr must have been returned by malloc/calloc/realloc, or be NULL"],
    analysis="realloc may move the memory block. If it moves, the old pointer is freed and the new pointer is returned. Newly allocated bytes (when growing) are uninitialized.",
))

register(LibraryFuncSummary(
    function_name="free",
    header="<cstdlib> / <stdlib.h>",
    signature="void free(void *ptr);",
    return_value_semantics="No return value.",
    side_effects="Deallocates the memory pointed to by ptr, making it unavailable.",
    writes_through_params=[],
    preconditions=["ptr must have been returned by malloc/calloc/realloc, or be NULL"],
    analysis="free does not modify the pointer itself, but the pointed-to memory is no longer valid.",
))

register(LibraryFuncSummary(
    function_name="strdup",
    header="<cstring> / <string.h>",
    signature="char *strdup(const char *s);",
    return_value_semantics="Returns a pointer to the duplicated string on success, NULL on failure.",
    side_effects="Allocates memory (via malloc) and copies s into it.",
    writes_through_params=[],
    preconditions=["s must be a null-terminated string"],
    analysis="strdup allocates and initializes memory in one call. The returned string is fully initialized.",
))

register(LibraryFuncSummary(
    function_name="ntohs",
    header="<arpa/inet.h>",
    signature="uint16_t ntohs(uint16_t netshort);",
    return_value_semantics="Returns the converted value in host byte order.",
    side_effects="None (pure function, no side effects).",
    writes_through_params=[],
    preconditions=[],
    analysis="ntohs is a pure function that converts a 16-bit value from network to host byte order. No side effects.",
))

register(LibraryFuncSummary(
    function_name="htons",
    header="<arpa/inet.h>",
    signature="uint16_t htons(uint16_t hostshort);",
    return_value_semantics="Returns the converted value in network byte order.",
    side_effects="None (pure function, no side effects).",
    writes_through_params=[],
    preconditions=[],
    analysis="htons is a pure function that converts a 16-bit value from host to network byte order. No side effects.",
))
