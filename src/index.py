import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_index(json_path: str | Path, out_dir: str | Path) -> Path:
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])

    tokenized = [_tokenize(c.get("text", "")) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    (out_dir / "meta.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out_dir


def search(out_dir: str | Path, query: str, k: int = 5, min_score: float = 0.0) -> list[dict]:
    out_dir = Path(out_dir)

    with open(out_dir / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    chunks = meta.get("chunks", [])

    q_tokens = _tokenize(query)
    if not q_tokens or not chunks:
        return []

    scores = bm25.get_scores(q_tokens)
    top_idx = np.argsort(scores)[::-1][:k]

    results: list[dict] = []
    rank = 1
    for idx in top_idx:
        idx = int(idx)
        s = float(scores[idx])
        if s <= min_score:
            continue

        ch = chunks[idx]
        results.append(
            {
                "rank": rank,
                "score": s,
                "page": ch.get("page"),
                "text": ch.get("text", ""),
            }
        )
        rank += 1

    return results
