import json
from pathlib import Path
from git import Repo
from repo_architect.config import get_logger

logger = get_logger(__name__)


def extract_commit_pairs(repo_path: str, output_file: str = "data/raw_commits.jsonl", max_commits: int = 400) -> int:
    repo = Repo(Path(repo_path).resolve())
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for commit in list(repo.iter_commits())[:max_commits]:
        if not commit.parents:
            continue
        msg = commit.message.strip()
        if len(msg) < 15 or msg.lower().startswith("merge "):
            continue
        try:
            diff = repo.git.diff(commit.parents[0].hexsha, commit.hexsha, unified=2)
        except Exception:
            continue
        if 80 <= len(diff) <= 3000:
            records.append({"instruction": "Write a concise technical commit summary for this git diff.", "input": diff, "output": msg})
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)
