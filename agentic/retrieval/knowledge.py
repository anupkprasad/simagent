"""Chunk + retrieve planner knowledge with the same hybrid embedder as tools."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from agentic.retrieval.embed import (
    blend_with_dense,
    dense_vectors,
    hashed_ngram_embed,
    hybrid_score,
)

logger = logging.getLogger(__name__)

INDEX_SCHEMA = "1.0"
INDEX_NAME = "knowledge_index.json"
_DEFAULT_CHUNK = 800
_DEFAULT_K = 8
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def default_index_path(knowledge_root: Path | str) -> Path:
    root = Path(knowledge_root)
    # Persist next to the knowledge tree: planner/knowledge_index.json
    return root.parent / INDEX_NAME if root.name == "knowledge" else root / INDEX_NAME


def _portable_path(path: str | Path) -> str:
    """Repository-relative path when the file is inside this checkout."""
    raw = Path(path)
    try:
        resolved = raw.expanduser().resolve()
    except OSError:
        return str(raw)
    root = Path(__file__).resolve().parents[2]
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def chunk_document(
    text: str,
    *,
    doc_key: str,
    category: str,
    source: str,
    max_chars: int = _DEFAULT_CHUNK,
) -> List[Dict[str, Any]]:
    """Split markdown/text on headings, then by size. Each chunk gets a cite id."""
    blob = (text or "").strip()
    if not blob:
        return []
    sections: List[tuple[str, str]] = []
    matches = list(_HEADING_RE.finditer(blob))
    if not matches:
        sections.append((doc_key.split(".")[-1], blob))
    else:
        preamble = blob[: matches[0].start()].strip()
        if preamble:
            sections.append((doc_key.split(".")[-1], preamble))
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(blob)
            body = blob[m.end() : end].strip()
            heading = m.group(2).strip()
            sections.append((heading, f"{m.group(0).strip()}\n{body}".strip()))

    chunks: List[Dict[str, Any]] = []
    for heading, body in sections:
        pieces = _split_size(body, max_chars)
        for i, piece in enumerate(pieces):
            slug = _slug(heading)
            cid = f"{doc_key}#{slug}" if i == 0 else f"{doc_key}#{slug}-{i + 1}"
            chunks.append(
                {
                    "id": cid,
                    "doc_key": doc_key,
                    "category": category,
                    "source": source,
                    "heading": heading,
                    "text": piece,
                    "embed_text": f"{doc_key} {heading} {piece}",
                }
            )
    return chunks


def _split_size(text: str, max_chars: int) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    parts: List[str] = []
    paragraphs = re.split(r"\n\s*\n", text)
    buf = ""
    for para in paragraphs:
        cand = f"{buf}\n\n{para}".strip() if buf else para
        if len(cand) <= max_chars:
            buf = cand
            continue
        if buf:
            parts.append(buf)
        if len(para) <= max_chars:
            buf = para
        else:
            for i in range(0, len(para), max_chars):
                parts.append(para[i : i + max_chars])
            buf = ""
    if buf:
        parts.append(buf)
    return parts


def _slug(heading: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (heading or "").lower()).strip("-")
    return s[:48] or "chunk"


def build_knowledge_chunks(loader: Any) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    docs = getattr(loader, "knowledge_docs", {}) or {}
    for doc_key, doc in docs.items():
        if (doc.get("name") or "").lower() == "readme":
            continue
        content = doc.get("content")
        if isinstance(content, dict):
            content = json.dumps(content, indent=2)
        chunks.extend(
            chunk_document(
                str(content or ""),
                doc_key=str(doc_key),
                category=str(doc.get("category") or "general"),
                source=_portable_path(doc.get("file_path") or ""),
            )
        )
    return chunks


def _file_fingerprint(loader: Any) -> Dict[str, float]:
    stamp: Dict[str, float] = {}
    for doc in (getattr(loader, "knowledge_docs", {}) or {}).values():
        path = doc.get("file_path")
        if path:
            stamp[_portable_path(path)] = _mtime(Path(path))
    return stamp


def load_or_build_index(
    loader: Any,
    *,
    index_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Return a persisted chunk index; rebuild when files change."""
    root = Path(getattr(loader, "knowledge_path", ".") or ".")
    path = Path(index_path) if index_path else default_index_path(root)
    fingerprint = _file_fingerprint(loader)
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("schema_version") == INDEX_SCHEMA
            and cached.get("fingerprint") == fingerprint
            and cached.get("chunks")
        ):
            return cached

    chunks = build_knowledge_chunks(loader)
    for ch in chunks:
        ch["embedding"] = hashed_ngram_embed(ch["embed_text"])
    payload = {
        "schema_version": INDEX_SCHEMA,
        "timestamp": datetime.now().isoformat(),
        "n_chunks": len(chunks),
        "fingerprint": fingerprint,
        "chunks": chunks,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote knowledge index %s chunks=%d", path, len(chunks))
    except OSError as exc:
        logger.warning("Could not persist knowledge index %s: %s", path, exc)
    return payload


def retrieve_knowledge_chunks(
    loader: Any,
    query: str,
    *,
    category: Optional[str] = None,
    top_k: int = _DEFAULT_K,
    index_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    index = load_or_build_index(loader, index_path=index_path)
    chunks = list(index.get("chunks") or [])
    if category:
        chunks = [c for c in chunks if c.get("category") == category]
    if not chunks:
        return []
    q = (query or "").strip() or "molecular dynamics protein simulation protocol"
    q_vec = hashed_ngram_embed(q)
    dense = dense_vectors(
        [q] + [c.get("embed_text") or c.get("text") or "" for c in chunks]
    )
    q_dense = dense[0] if dense else None
    d_dense = dense[1:] if len(dense) > 1 else []

    scored: List[tuple[float, Dict[str, Any]]] = []
    for i, ch in enumerate(chunks):
        d_vec = ch.get("embedding") or hashed_ngram_embed(ch.get("embed_text") or "")
        score = hybrid_score(q, ch.get("embed_text") or ch.get("text") or "", q_vec, d_vec)
        doc_dense = d_dense[i] if i < len(d_dense) else None
        score = blend_with_dense(score, q_dense, doc_dense)
        scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ch for _, ch in scored[: max(1, int(top_k))]]


def format_knowledge_chunks(
    chunks: Iterable[Dict[str, Any]],
    *,
    max_chars: int = 6000,
) -> str:
    lines = [
        "**Retrieved knowledge** (top chunks — cite the [kb:…] ids, do not dump whole manuals):",
        "",
    ]
    used = 0
    for ch in chunks:
        cid = ch.get("id") or "unknown"
        heading = ch.get("heading") or ""
        text = (ch.get("text") or "").strip()
        block = f"[kb:{cid}] {heading}\n{text}\n"
        if used + len(block) > max_chars and used:
            lines.append("…[additional matching chunks omitted]")
            break
        lines.append(block)
        used += len(block)
    if used == 0:
        return ""
    return "\n".join(lines).strip()


def retrieve_knowledge_for_prompt(
    loader: Any,
    query: str,
    *,
    category: Optional[str] = None,
    top_k: int = _DEFAULT_K,
    max_chars: int = 6000,
    index_path: Optional[Path | str] = None,
    persist_copy: Optional[Path | str] = None,
) -> str:
    """Planner-ready retrieved chunks + citations (not a full-file dump)."""
    chunks = retrieve_knowledge_chunks(
        loader,
        query,
        category=category,
        top_k=top_k,
        index_path=index_path,
    )
    if persist_copy:
        try:
            src = Path(index_path) if index_path else default_index_path(
                Path(getattr(loader, "knowledge_path", ".") or ".")
            )
            dest = Path(persist_copy)
            if src.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError as exc:
            logger.debug("Could not copy knowledge index to %s: %s", persist_copy, exc)
    formatted = format_knowledge_chunks(chunks, max_chars=max_chars)
    if formatted:
        return formatted
    return (
        "**Retrieved knowledge:** no matching chunks. "
        "Use tool manuals and standard MD practice."
    )
