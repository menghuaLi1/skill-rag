from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from llama_index.core import Document, Settings as LlamaSettings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

from config import get_settings


class ExperienceMemory:
    def __init__(self) -> None:
        self.base_dir: Path | None = None
        self._entries: list[dict[str, Any]] = []
        self._index: VectorStoreIndex | None = None

    def configure(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_entries()
        self._load_index()

    @property
    def _storage_dir(self) -> Path:
        if self.base_dir is None:
            raise RuntimeError("ExperienceMemory is not configured")
        return self.base_dir / "storage" / "experience_memory"

    @property
    def _entries_path(self) -> Path:
        return self._storage_dir / "entries.json"

    @property
    def _vector_dir(self) -> Path:
        return self._storage_dir / "vector"

    @property
    def _conflicts_path(self) -> Path:
        return self._storage_dir / "conflicts.json"

    def _supports_embeddings(self) -> bool:
        return bool(get_settings().embedding_api_key)

    def _build_embed_model(self) -> OpenAIEmbedding:
        settings = get_settings()
        return OpenAIEmbedding(
            api_key=settings.embedding_api_key,
            api_base=settings.embedding_base_url,
            model=settings.embedding_model,
        )

    def _load_entries(self) -> None:
        if not self._entries_path.exists():
            self._entries = []
            return
        try:
            payload = json.loads(self._entries_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._entries = []
            return
        if not isinstance(payload, list):
            self._entries = []
            return
        self._entries = [item for item in payload if isinstance(item, dict)]

    def _save_entries(self) -> None:
        self._entries_path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_index(self) -> None:
        if not self._supports_embeddings():
            self._index = None
            return
        if not self._vector_dir.exists() or not list(self._vector_dir.glob("*")):
            self._index = None
            return
        try:
            LlamaSettings.embed_model = self._build_embed_model()
            storage_context = StorageContext.from_defaults(persist_dir=str(self._vector_dir))
            self._index = load_index_from_storage(storage_context)
        except Exception:
            self._index = None

    def _rebuild_index(self) -> None:
        if not self._supports_embeddings():
            self._index = None
            return

        alive_entries = [item for item in self._entries if not self._is_expired(item)]
        if not alive_entries:
            self._index = None
            return

        try:
            LlamaSettings.embed_model = self._build_embed_model()
            splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)
            docs = [
                Document(
                    text=str(item.get("text", "")),
                    metadata={
                        "source": "memory/experience",
                        "experience_id": str(item.get("id", "")),
                        "created_at": float(item.get("created_at", 0.0) or 0.0),
                    },
                )
                for item in alive_entries
                if str(item.get("text", "")).strip()
            ]
            if not docs:
                self._index = None
                return
            nodes = splitter.get_nodes_from_documents(docs)
            self._index = VectorStoreIndex(nodes)
            self._index.storage_context.persist(persist_dir=str(self._vector_dir))
        except Exception:
            self._index = None

    def _is_expired(self, item: dict[str, Any]) -> bool:
        expires_at = float(item.get("expires_at", 0.0) or 0.0)
        return expires_at > 0 and expires_at <= time.time()

    def cleanup_expired(self) -> int:
        before = len(self._entries)
        self._entries = [item for item in self._entries if not self._is_expired(item)]
        removed = before - len(self._entries)
        if removed > 0:
            self._save_entries()
            self._rebuild_index()
        return removed

    def _append_conflict(self, conflict: dict[str, Any]) -> None:
        existing: list[dict[str, Any]] = []
        if self._conflicts_path.exists():
            try:
                payload = json.loads(self._conflicts_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = []
            if isinstance(payload, list):
                existing = [item for item in payload if isinstance(item, dict)]
        existing.append(conflict)
        self._conflicts_path.write_text(
            json.dumps(existing[-200:], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_experience(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_message: str,
        tool_calls: list[dict[str, Any]] | None = None,
        retrieval_steps: list[dict[str, Any]] | None = None,
        ttl_days: int = 180,
    ) -> bool:
        user_text = user_message.strip()
        assistant_text = assistant_message.strip()
        if not user_text or not assistant_text:
            return False

        tool_names = [str(item.get("tool", "")).strip() for item in (tool_calls or []) if str(item.get("tool", "")).strip()]
        retrieval_stages = [str(item.get("stage", "")).strip() for item in (retrieval_steps or []) if str(item.get("stage", "")).strip()]
        experience_text = (
            f"用户问题: {user_text}\n"
            f"执行摘要: 使用工具 {', '.join(tool_names) if tool_names else '无'}; "
            f"检索阶段 {', '.join(retrieval_stages) if retrieval_stages else '无'}\n"
            f"回答结果: {assistant_text[:800]}"
        )
        fingerprint = hashlib.md5(experience_text.encode("utf-8")).hexdigest()
        if any(str(item.get("fingerprint", "")) == fingerprint for item in self._entries):
            return False

        now = time.time()
        expires_at = now + max(1, ttl_days) * 24 * 3600
        experience_id = hashlib.md5(f"{session_id}:{now}:{fingerprint}".encode("utf-8")).hexdigest()

        # Conflict heuristic: same normalized question stem but opposite outcomes.
        normalized_q = " ".join(user_text.lower().split())[:120]
        outcome = "failed" if any(token in assistant_text.lower() for token in ["失败", "无法", "没有找到", "error"]) else "success"
        for item in self._entries:
            if str(item.get("normalized_q", "")) != normalized_q:
                continue
            if str(item.get("outcome", "")) != outcome:
                self._append_conflict(
                    {
                        "created_at": now,
                        "session_id": session_id,
                        "normalized_q": normalized_q,
                        "previous_outcome": item.get("outcome"),
                        "current_outcome": outcome,
                        "previous_id": item.get("id"),
                        "current_id": experience_id,
                    }
                )
                break

        self._entries.append(
            {
                "id": experience_id,
                "session_id": session_id,
                "created_at": now,
                "expires_at": expires_at,
                "fingerprint": fingerprint,
                "normalized_q": normalized_q,
                "outcome": outcome,
                "text": experience_text,
            }
        )
        self._save_entries()
        self._rebuild_index()
        return True

    def retrieve(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        self.cleanup_expired()
        if not self._supports_embeddings():
            return []
        if self._index is None:
            self._load_index()
        if self._index is None:
            return []

        try:
            retriever = self._index.as_retriever(similarity_top_k=top_k)
            results = retriever.retrieve(query)
        except Exception:
            return []

        payload: list[dict[str, Any]] = []
        for item in results:
            node = getattr(item, "node", item)
            text = getattr(node, "text", "") or getattr(node, "get_content", lambda: "")()
            payload.append(
                {
                    "text": str(text).strip(),
                    "score": float(getattr(item, "score", 0.0) or 0.0),
                    "source": "memory/experience",
                }
            )
        return payload


experience_memory = ExperienceMemory()
