from pathlib import Path
from repo_architect.ingestion.ast_parser import parse_repository


def test_ast_parser_extracts_functions_and_classes(tmp_path: Path):
    sample_py = tmp_path / "service.py"
    sample_py.write_text("class AuthService:\n    def verify_token(self, token: str) -> bool:\n        return len(token) > 10\n\ndef helper_fn():\n    return 42\n", encoding="utf-8")
    chunks = parse_repository(str(tmp_path), repo_id="test-repo")
    assert len(chunks) == 3
    assert all(c["metadata"]["file_path"] == "service.py" for c in chunks)
