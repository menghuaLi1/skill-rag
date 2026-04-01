from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from llama_index.core import Document, Settings as LlamaSettings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding

from config import get_settings
from knowledge_retrieval.mineru_preprocessor import mineru_preprocessor
from knowledge_retrieval.types import Evidence, IndexStatus


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
ALNUM_PATTERN = re.compile(r"[A-Za-z0-9_]+")
CHINESE_BLOCK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


class KnowledgeIndexer:
    def __init__(self) -> None:
        self.base_dir: Path | None = None
        self._vector_index: VectorStoreIndex | None = None
        self._documents: list[dict[str, Any]] = []
        self._source_state: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._rebuild_timer_lock = threading.Lock()
        self._rebuild_timer: threading.Timer | None = None
        self._building = False
        self._last_built_at: float | None = None
        self._avg_doc_length = 0.0
        self._document_frequencies: Counter[str] = Counter()
        self._vector_ready = False
        self._bm25_ready = False

    def configure(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        mineru_preprocessor.configure(base_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._vector_dir.mkdir(parents=True, exist_ok=True)
        self._bm25_dir.mkdir(parents=True, exist_ok=True)
        self._derived_dir.mkdir(parents=True, exist_ok=True)
        self._load_manifest()
        self._load_vector_index()

    @property
    def _knowledge_dir(self) -> Path:
        if self.base_dir is None:
            raise RuntimeError("KnowledgeIndexer is not configured")
        return self.base_dir / "knowledge"

    @property
    def _storage_dir(self) -> Path:
        if self.base_dir is None:
            raise RuntimeError("KnowledgeIndexer is not configured")
        return self.base_dir / "storage" / "knowledge"

    @property
    def _manifest_path(self) -> Path:
        return self._storage_dir / "manifest.json"

    @property
    def _vector_dir(self) -> Path:
        return self._storage_dir / "vector"

    @property
    def _bm25_dir(self) -> Path:
        return self._storage_dir / "bm25"

    @property
    def _derived_dir(self) -> Path:
        return self._storage_dir / "derived"

    def _supports_embeddings(self) -> bool:
        return bool(get_settings().embedding_api_key)

    def _build_embed_model(self) -> OpenAIEmbedding:
        settings = get_settings()
        return OpenAIEmbedding(
            api_key=settings.embedding_api_key,
            api_base=settings.embedding_base_url,
            model=settings.embedding_model,
        )

    def status(self) -> IndexStatus:
        return IndexStatus(
            ready=bool(self._documents) and (self._vector_ready or self._bm25_ready),
            building=self._building,
            last_built_at=self._last_built_at,
            indexed_files=len({item["source_path"] for item in self._documents}),
            vector_ready=self._vector_ready,
            bm25_ready=self._bm25_ready,
        )

    def is_building(self) -> bool:
        return self._building

    def rebuild_index(self) -> None:
        if self.base_dir is None:
            return

        with self._lock:
            self._building = True
            try:
                self._documents, self._source_state, changed = self._build_documents_incremental()
                if changed:
                    self._write_manifest()
                self._prepare_bm25_stats()
                should_rebuild_vector = changed or not self._vector_ready
                if should_rebuild_vector:
                    self._build_vector_index()
                self._last_built_at = time.time()
            finally:
                self._building = False

    def request_rebuild_debounced(self, *, debounce_seconds: float = 2.0) -> None:
        if self.base_dir is None:
            return

        delay = max(0.0, float(debounce_seconds))
        with self._rebuild_timer_lock:
            if self._rebuild_timer is not None:
                self._rebuild_timer.cancel()

            timer = threading.Timer(delay, self._run_debounced_rebuild)
            timer.daemon = True
            self._rebuild_timer = timer
            timer.start()

    def _run_debounced_rebuild(self) -> None:
        try:
            self.rebuild_index()
        finally:
            with self._rebuild_timer_lock:
                self._rebuild_timer = None

    def _relative_path(self, path: Path) -> str:
        if self.base_dir is None:
            return str(path)
        return str(path.relative_to(self.base_dir)).replace("\\", "/")

    def _build_documents_incremental(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], bool]:
        if not self._knowledge_dir.exists():
            changed = bool(self._documents or self._source_state)
            return [], {}, changed

        previous_state = dict(self._source_state)
        previous_docs_by_source: dict[str, list[dict[str, Any]]] = {}
        for item in self._documents:
            source_path = str(item.get("source_path", "")).strip()
            if not source_path:
                continue
            previous_docs_by_source.setdefault(source_path, []).append(item)

        sources: dict[str, dict[str, Any]] = {}
        for path in sorted(self._knowledge_dir.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in {".md", ".txt", ".json"}:
                continue
            source_path = self._relative_path(path)
            stat = path.stat()
            sources[source_path] = {
                "kind": "native",
                "suffix": suffix,
                "input_path": path,
                "source_path": source_path,
                "source_type": suffix.lstrip("."),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }

        derived_markdowns = mineru_preprocessor.preprocess()
        for path in sorted(derived_markdowns):
            if not path.is_file():
                continue
            metadata = self._load_derived_metadata(path)
            source_path = str(metadata.get("source_path", self._relative_path(path))).strip()
            if not source_path:
                continue
            source_type = str(metadata.get("source_type", "pdf")).strip() or "pdf"
            source_abs = self.base_dir / source_path if self.base_dir is not None else path
            if source_abs.exists() and source_abs.is_file():
                stat = source_abs.stat()
            else:
                stat = path.stat()
            sources[source_path] = {
                "kind": "derived",
                "suffix": ".md",
                "input_path": path,
                "source_path": source_path,
                "source_type": source_type,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }

        next_state: dict[str, dict[str, Any]] = {}
        next_docs_by_source: dict[str, list[dict[str, Any]]] = {}
        changed = False

        for source_path in sorted(sources.keys()):
            entry = sources[source_path]
            signature = {
                "kind": str(entry["kind"]),
                "source_type": str(entry["source_type"]),
                "mtime": float(entry["mtime"]),
                "size": int(entry["size"]),
            }
            next_state[source_path] = signature

            previous_signature = previous_state.get(source_path)
            cached_docs = previous_docs_by_source.get(source_path)
            if previous_signature == signature and cached_docs:
                next_docs_by_source[source_path] = cached_docs
                continue

            changed = True
            input_path = Path(entry["input_path"])
            suffix = str(entry["suffix"])
            source_type = str(entry["source_type"])

            if suffix == ".md":
                chunks = self._split_markdown(
                    input_path,
                    source_path_override=source_path,
                    source_type_override=source_type,
                )
            elif suffix == ".txt":
                chunks = self._split_plain_text(input_path)
            elif suffix == ".json":
                chunks = self._split_json(input_path)
            else:
                chunks = []

            next_docs_by_source[source_path] = chunks

        if set(previous_state.keys()) != set(next_state.keys()):
            changed = True

        documents: list[dict[str, Any]] = []
        for source_path in sorted(next_docs_by_source.keys()):
            documents.extend(next_docs_by_source[source_path])
        return documents, next_state, changed

    def _split_markdown(
        self,
        path: Path,
        *,
        source_path_override: str | None = None,
        source_type_override: str | None = None,
    ) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8")
        source_path = source_path_override or self._relative_path(path)
        source_type = source_type_override or "md"
        sections: list[tuple[list[str], list[str]]] = []
        heading_stack: list[str] = []
        current_lines: list[str] = []

        def flush_section() -> None:
            if not current_lines:
                return
            heading_path = heading_stack[:] if heading_stack else [path.stem]
            sections.append((heading_path, current_lines[:]))

        for raw_line in text.splitlines():
            match = HEADING_PATTERN.match(raw_line)
            if not match:
                current_lines.append(raw_line)
                continue

            flush_section()
            current_lines = [raw_line]
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)

        flush_section()
        if not sections:
            sections = [([path.stem], text.splitlines())]

        chunks: list[dict[str, Any]] = []
        for section_index, (heading_path, lines) in enumerate(sections, start=1):
            section_text = "\n".join(lines).strip()
            if not section_text:
                continue
            parent_id = f"{source_path}::{' > '.join(heading_path)}"
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", section_text) if part.strip()]
            if not paragraphs:
                paragraphs = [section_text]

            for paragraph_index, paragraph in enumerate(paragraphs, start=1):
                content = paragraph.strip()
                if not content:
                    continue
                slices = [content[index : index + 1200] for index in range(0, len(content), 1200)] or [content]
                for slice_index, slice_text in enumerate(slices, start=1):
                    locator = f"{' > '.join(heading_path)} / 段落 {paragraph_index}"
                    if len(slices) > 1:
                        locator = f"{locator}.{slice_index}"
                    chunks.append(
                        {
                            "doc_id": f"{parent_id}::child::{paragraph_index}.{slice_index}",
                            "parent_id": parent_id,
                            "source_path": source_path,
                            "source_type": source_type,
                            "locator": locator,
                            "text": slice_text,
                            "parent_text": section_text,
                            "section_index": section_index,
                        }
                    )
        return chunks

    def _split_plain_text(self, path: Path) -> list[dict[str, Any]]:
        source_path = self._relative_path(path)
        source_type = path.suffix.lower().lstrip(".") or "txt"
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []

        parent_id = f"{source_path}::text"
        chunks: list[dict[str, Any]] = []
        slices = [text[index : index + 1200] for index in range(0, len(text), 1200)] or [text]
        for slice_index, slice_text in enumerate(slices, start=1):
            chunks.append(
                {
                    "doc_id": f"{parent_id}::child::{slice_index}",
                    "parent_id": parent_id,
                    "source_path": source_path,
                    "source_type": source_type,
                    "locator": f"文本片段 {slice_index}",
                    "text": slice_text,
                    "parent_text": text,
                }
            )
        return chunks

    def _load_derived_metadata(self, path: Path) -> dict[str, Any]:
        metadata_path = path.with_suffix(f"{path.suffix}.meta.json")
        if not metadata_path.exists():
            return {}
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def _split_json(self, path: Path) -> list[dict[str, Any]]:
        source_path = self._relative_path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []

        chunks: list[dict[str, Any]] = []
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            label = str(item.get("label", "")).strip()
            url = str(item.get("url", "")).strip()
            if not question and not answer:
                continue

            record_id = str(item.get("record_id") or item.get("id") or index)
            locator = f"记录 {record_id}"
            parts = []
            if question:
                parts.append(f"Question: {question}")
            if answer:
                parts.append(f"Answer: {answer}")
            if label:
                parts.append(f"Label: {label}")
            if url:
                parts.append(f"URL: {url}")
            text = "\n".join(parts)
            parent_id = f"{source_path}::record::{record_id}"
            chunks.append(
                {
                    "doc_id": f"{parent_id}::child::1",
                    "parent_id": parent_id,
                    "source_path": source_path,
                    "source_type": "json",
                    "locator": locator,
                    "text": text,
                    "parent_text": text,
                    "record_id": record_id,
                }
            )
        return chunks

    def _write_manifest(self) -> None:
        payload = {
            "built_at": time.time(),
            "documents": self._documents,
            "source_state": self._source_state,
        }
        self._manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_manifest(self) -> None:
        if not self._manifest_path.exists():
            self._documents = []
            self._source_state = {}
            self._bm25_ready = False
            return
        try:
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._documents = []
            self._source_state = {}
            self._bm25_ready = False
            return
        self._documents = list(payload.get("documents", []))
        source_state = payload.get("source_state", {})
        if isinstance(source_state, dict):
            self._source_state = {
                str(key): value
                for key, value in source_state.items()
                if isinstance(key, str) and isinstance(value, dict)
            }
        else:
            self._source_state = {}
        self._last_built_at = payload.get("built_at")
        self._prepare_bm25_stats()

    def _prepare_bm25_stats(self) -> None:
        if not self._documents:
            self._avg_doc_length = 0.0
            self._document_frequencies = Counter()
            self._bm25_ready = False
            return

        self._document_frequencies = Counter()
        doc_lengths: list[int] = []
        for item in self._documents:
            tokens = self._tokenize(str(item.get("text", "")))
            item["tokens"] = tokens
            doc_lengths.append(len(tokens))
            for token in set(tokens):
                self._document_frequencies[token] += 1

        self._avg_doc_length = sum(doc_lengths) / max(1, len(doc_lengths))
        self._bm25_ready = True

    def _build_vector_index(self) -> None:
        if not self._supports_embeddings() or not self._documents:
            self._vector_index = None
            self._vector_ready = False
            return

        try:
            LlamaSettings.embed_model = self._build_embed_model()
            documents = [
                Document(
                    text=str(item["text"]),
                    metadata={
                        "doc_id": item["doc_id"],
                        "parent_id": item["parent_id"],
                        "source_path": item["source_path"],
                        "source_type": item["source_type"],
                        "locator": item["locator"],
                    },
                )
                for item in self._documents
            ]
            self._vector_index = VectorStoreIndex.from_documents(documents)
            self._vector_index.storage_context.persist(persist_dir=str(self._vector_dir))
            self._vector_ready = True
        except Exception:
            self._vector_index = None
            self._vector_ready = False

    def _load_vector_index(self) -> None:
        if not self._supports_embeddings():
            self._vector_index = None
            self._vector_ready = False
            return
        if not list(self._vector_dir.glob("*")):
            self._vector_index = None
            self._vector_ready = False
            return
        try:
            LlamaSettings.embed_model = self._build_embed_model()
            storage_context = StorageContext.from_defaults(persist_dir=str(self._vector_dir))
            self._vector_index = load_index_from_storage(storage_context)
            self._vector_ready = True
        except Exception:
            self._vector_index = None
            self._vector_ready = False

    def _ensure_loaded(self) -> None:
        if not self._documents:
            self._load_manifest()
        if self._vector_index is None and self._supports_embeddings():
            self._load_vector_index()

    def _matches_path_filters(self, source_path: str, path_filters: list[str] | None) -> bool:
        if not path_filters:
            return True
        normalized = source_path.replace("\\", "/")
        for path_filter in path_filters:
            candidate = path_filter.replace("\\", "/").strip()
            if not candidate:
                continue
            if normalized == candidate or normalized.startswith(f"{candidate}/"):
                return True
        return False

    def retrieve_vector(
        self,
        query: str,
        *,
        top_k: int = 4,
        path_filters: list[str] | None = None,
    ) -> list[Evidence]:
        self._ensure_loaded()
        if self._vector_index is None:
            return []

        retriever = self._vector_index.as_retriever(similarity_top_k=max(top_k * 4, top_k))
        try:
            results = retriever.retrieve(query)
        except Exception:
            return []

        payload: list[Evidence] = []
        for item in results:
            node = getattr(item, "node", item)
            metadata = getattr(node, "metadata", {}) or {}
            source_path = str(metadata.get("source_path", ""))
            if not self._matches_path_filters(source_path, path_filters):
                continue
            text = getattr(node, "text", "") or getattr(node, "get_content", lambda: "")()
            raw_parent_id = metadata.get("parent_id")
            parent_id = str(raw_parent_id).strip() if raw_parent_id else None
            payload.append(
                Evidence(
                    source_path=source_path,
                    source_type=str(metadata.get("source_type", "unknown")),
                    locator=str(metadata.get("locator", "")),
                    snippet=str(text).strip(),
                    channel="vector",
                    score=float(getattr(item, "score", 0.0) or 0.0),
                    parent_id=parent_id,
                )
            )
            if len(payload) >= top_k:
                break
        return payload

    def retrieve_bm25(
        self,
        query: str,
        *,
        top_k: int = 4,
        path_filters: list[str] | None = None,
        query_hints: list[str] | None = None,
    ) -> list[Evidence]:
        self._ensure_loaded()
        if not self._documents or not self._bm25_ready:
            return []

        hints = " ".join(query_hints or [])
        query_tokens = self._tokenize(f"{query} {hints}".strip())
        if not query_tokens:
            return []

        candidates = [
            item for item in self._documents if self._matches_path_filters(str(item["source_path"]), path_filters)
        ]
        if not candidates:
            candidates = list(self._documents)

        scores: list[tuple[dict[str, Any], float]] = []
        corpus_size = max(1, len(self._documents))
        k1 = 1.5
        b = 0.75
        for item in candidates:
            doc_tokens = item.get("tokens", [])
            if not doc_tokens:
                continue
            token_counts = Counter(doc_tokens)
            doc_len = len(doc_tokens)
            score = 0.0
            for token in query_tokens:
                if token not in token_counts:
                    continue
                df = self._document_frequencies.get(token, 0)
                if df <= 0:
                    continue
                idf = math.log(1 + ((corpus_size - df + 0.5) / (df + 0.5)))
                freq = token_counts[token]
                denominator = freq + k1 * (1 - b + b * (doc_len / max(1.0, self._avg_doc_length)))
                score += idf * ((freq * (k1 + 1)) / max(denominator, 1e-9))
            if score > 0:
                scores.append((item, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        payload: list[Evidence] = []
        for item, score in scores[:top_k]:
            raw_parent_id = item.get("parent_id")
            parent_id = str(raw_parent_id).strip() if raw_parent_id else None
            payload.append(
                Evidence(
                    source_path=str(item["source_path"]),
                    source_type=str(item["source_type"]),
                    locator=str(item["locator"]),
                    snippet=str(item["text"]).strip(),
                    channel="bm25",
                    score=score,
                    parent_id=parent_id,
                )
            )
        return payload

    def _tokenize(self, text: str) -> list[str]:
        lowered = text.lower()
        tokens: list[str] = []
        tokens.extend(ALNUM_PATTERN.findall(lowered))
        for match in CHINESE_BLOCK_PATTERN.findall(lowered):
            tokens.extend(list(match))
            if len(match) > 1:
                tokens.extend(match[index : index + 2] for index in range(len(match) - 1))
        return tokens


knowledge_indexer = KnowledgeIndexer()
