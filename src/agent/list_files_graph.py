from typing import Any, TypedDict
from pathlib import Path

from langgraph.graph import StateGraph


ISO_TYPE = {
    "BHKN": "Bài học kinh nghiệm",
    "ISO": "Tài liệu ISO",
    "HDCV": "Hướng dẫn công việc",
    "TTQT": "Thủ tục & Quy trình",
    "QĐ": "Quy định",
    "CS": "Chính sách",
    "MTCL": "Mục tiêu chất lượng",
    "MTCV": "Mô tả công việc",
    "QH": "Quyền hạn",
    "ST": "Sổ tay văn hóa",
}


class StateSchema(TypedDict):
    pass


class OutputSchema(TypedDict):
    nodes: list[dict[str, Any]] | None


def build_tree(path: Path) -> dict[str, Any]:
    if (path.is_file() and path.suffix in [".docx"]) or (
        path.is_dir() and path.name == "BHKN"
    ):
        return {"label": path.name, "value": path.name}
    node = {
        "label": ISO_TYPE.get(path.name, path.name),
        "value": path.name,
        "children": [],
    }
    for child in sorted(path.iterdir()):
        if child.is_file() and child.suffix not in [".docx"]:
            continue
        node["children"].append(build_tree(child))
    return node


def list_files_in_directory(state: StateSchema) -> dict[str, Any]:
    try:
        root = Path("data")
        nodes = [build_tree(child) for child in sorted(root.iterdir())]
        return {"nodes": nodes}
    except Exception as e:
        print(str(e))
        return {"nodes": []}


graph = (
    StateGraph(state_schema=StateSchema, output_schema=OutputSchema)
    # define nodes
    .add_node("list_files_in_directory", list_files_in_directory)
    # define workflow
    .set_entry_point("list_files_in_directory")
    .set_finish_point("list_files_in_directory")
    # compile the graph
    .compile(name="main_graph")
)
