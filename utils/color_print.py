"""
Color print utilities using ANSI escape codes.
No external dependencies required.

Usage:
    from utils.color_print import cprint, C

    cprint("Hello!", C.GREEN, C.BOLD)
    cprint("Warning", C.YELLOW, end="")
"""


class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[32m"
    RED     = "\033[31m"
    YELLOW  = "\033[33m"
    CYAN    = "\033[36m"
    MAGENTA = "\033[35m"


def cprint(text: str, *codes: str, end: str = "\n"):
    """Print text with ANSI color codes.

    Args:
        text:  The text to print.
        codes: Any number of C.* constants to apply.
        end:   Line ending (same as built-in print).
    """
    print("".join(codes) + text + C.RESET, end=end)
