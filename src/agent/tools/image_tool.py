from pathlib import Path

from langchain_core.tools import tool
from agent.utils.logger_handler import get_logger
from agent.utils.path_handler import get_project_root

logger = get_logger()

_IMG_TYPES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"}

_image_cache: dict[str, Path | None] = {}


def _find_image(filename: str) -> Path | None:
    if filename in _image_cache:
        return _image_cache[filename]

    data_dir = Path(get_project_root()) / "data"
    if not data_dir.exists():
        _image_cache[filename] = None
        return None

    for match in data_dir.rglob(filename):
        _image_cache[filename] = match
        return match

    _image_cache[filename] = None
    return None


@tool("view_image")
def view_image(image_path: str) -> str:
    """
    查看知识库中引用的图片信息。
    当 knowledge_base_search 或 fetch_neighbor_context 返回的结果中包含图片路径时，
    使用此工具查看图片的详细信息（文件路径、大小、是否存在等）。

    Args:
        image_path: 图片的相对路径，从检索结果的 image_paths 字段获取。
            格式如 "images/xxx.jpg"。

    Returns:
        图片的详细信息，包括绝对路径、文件大小和存在状态。
    """
    filename = Path(image_path).name
    abs_path = _find_image(filename)

    if abs_path is None:
        return (
            f"图片不存在: {image_path}\n"
            f"在 data/ 目录下未找到文件 '{filename}'。"
        )

    size_bytes = abs_path.stat().st_size
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

    suffix = abs_path.suffix.lower()
    is_image = suffix in _IMG_TYPES

    return (
        f"图片信息:\n"
        f"  路径: {abs_path}\n"
        f"  大小: {size_str}\n"
        f"  类型: {suffix}\n"
        f"  是图片: {is_image}"
    )
