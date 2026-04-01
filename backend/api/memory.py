from __future__ import annotations

from fastapi import APIRouter

from graph.experience_memory import experience_memory
from graph.memory_indexer import memory_indexer

router = APIRouter()


@router.post("/memory/maintenance")
async def run_memory_maintenance() -> dict[str, int | bool]:
    removed = experience_memory.cleanup_expired()
    memory_indexer.rebuild_index()
    return {
        "ok": True,
        "expired_experience_removed": removed,
    }
