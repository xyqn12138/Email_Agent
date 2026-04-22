"""
路径处理工具类,保证项目中路径的统一
"""
from pathlib import Path

def get_project_root() -> str:
    """
    获取项目根路径。

    Returns:
        项目根路径字符串
    """
    return Path(__file__).resolve().parents[3]

def get_absolute_path(relative_path: str) -> str:
    """
    获取相对路径对应的绝对路径。

    Args:
        relative_path: 相对路径字符串
        
    Returns:
        绝对路径字符串
    """
    return Path(get_project_root()) / relative_path

if __name__ == "__main__":
    path = get_absolute_path(r"src\agent\Loader\doc_loader.py")
    print(path)