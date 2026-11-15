"""Source/output separation without loading scientific dependencies."""
from pathlib import Path

from .errors import CountingError


def output_path(path, source):
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise CountingError("invalid_output_path", "输出必须使用绝对路径。")
    target = target.resolve()
    source = Path(source).resolve()
    if target == source or source.parent == target.parent or source.parent in target.parents:
        raise CountingError("source_directory_read_only", "输出必须位于原图目录之外。")
    return target
