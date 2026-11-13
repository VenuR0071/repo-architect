import hashlib
from pathlib import Path
from typing import Dict, List
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser
from repo_architect.config import get_logger, settings

logger = get_logger(__name__)

LANGUAGES: Dict[str, Language] = {
    ".py": Language(tspython.language()),
    ".java": Language(tsjava.language()),
    ".js": Language(tsjs.language()),
}

TARGET_NODE_TYPES = {"function_definition", "class_definition", "method_declaration", "class_declaration", "function_declaration"}
IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "dist", "build", ".idea", ".vscode"}


def compute_chunk_id(repo_id: str, posix_path: str, start_line: int, code: str) -> str:
    raw = f"{repo_id}:{posix_path}:{start_line}:{code}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def parse_file(file_path: Path, repo_root: Path, repo_id: str) -> List[Dict]:
    ext = file_path.suffix.lower()
    if ext not in LANGUAGES:
        return []
    parser = Parser(LANGUAGES[ext])
    try:
        code_bytes = file_path.read_bytes()
    except OSError as exc:
        logger.warning(f"Skipping unreadable file {file_path}: {exc}")
        return []

    tree = parser.parse(code_bytes)
    posix_path = file_path.relative_to(repo_root).as_posix()
    chunks: List[Dict] = []

    def walk(node):
        if node.type in TARGET_NODE_TYPES:
            snippet = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
            if snippet:
                truncated = snippet[: settings.chunk_max_chars]
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                chunks.append({
                    "id": compute_chunk_id(repo_id, posix_path, start_line, truncated),
                    "text": truncated,
                    "metadata": {
                        "repo_id": repo_id,
                        "file_path": posix_path,
                        "start_line": start_line,
                        "end_line": end_line,
                        "node_type": node.type,
                        "extension": ext,
                    },
                })
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return chunks


def parse_repository(repo_path: str, repo_id: str) -> List[Dict]:
    root = Path(repo_path).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Repository directory does not exist: {root}")
    all_chunks: List[Dict] = []
    for path in root.rglob("*"):
        if any(part in IGNORED_DIRS or part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            all_chunks.extend(parse_file(path, root, repo_id))
    logger.info(f"Parsed {len(all_chunks)} AST blocks from {root}")
    return all_chunks
