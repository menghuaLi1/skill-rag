from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path

from config import get_settings


class MineruPreprocessor:
    def __init__(self) -> None:
        self.base_dir: Path | None = None

    def configure(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _knowledge_dir(self) -> Path:
        if self.base_dir is None:
            raise RuntimeError("MineruPreprocessor is not configured")
        return self.base_dir / "knowledge"

    @property
    def _output_dir(self) -> Path:
        if self.base_dir is None:
            raise RuntimeError("MineruPreprocessor is not configured")
        return self.base_dir / "storage" / "knowledge" / "derived" / "mineru"

    @property
    def _manifest_path(self) -> Path:
        return self._output_dir / "manifest.json"

    def preprocess(self) -> list[Path]:
        settings = get_settings()
        if self.base_dir is None or not settings.mineru_enabled:
            return []
        if not self._knowledge_dir.exists():
            return []

        previous_manifest = self._read_manifest()
        next_manifest: dict[str, dict[str, str | float]] = {}
        derived_markdown_files: list[Path] = []

        for source_path in sorted(self._knowledge_dir.rglob("*.pdf")):
            if not source_path.is_file():
                continue

            source_rel = self._relative_path(source_path)
            source_mtime = source_path.stat().st_mtime
            previous_entry = previous_manifest.get(source_rel, {})
            derived_rel = str(previous_entry.get("derived_markdown", "")).strip()
            derived_abs = (self.base_dir / derived_rel).resolve() if derived_rel and self.base_dir else None

            needs_refresh = bool(source_mtime > float(previous_entry.get("source_mtime", 0.0)))

            if needs_refresh:
                built_path = self._build_markdown_from_pdf(source_path)
                if built_path is not None:
                    derived_abs = built_path

            if derived_abs is None or not derived_abs.exists():
                continue

            derived_markdown_files.append(derived_abs)
            next_manifest[source_rel] = {
                "source_mtime": source_mtime,
                "derived_markdown": self._relative_path(derived_abs),
                "updated_at": time.time(),
            }

        self._manifest_path.write_text(
            json.dumps(next_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return derived_markdown_files

    def _build_markdown_from_pdf(self, source_path: Path) -> Path | None:
        settings = get_settings()
        if self.base_dir is None:
            return None

        relative_stem = source_path.relative_to(self._knowledge_dir).with_suffix("")
        run_output_dir = self._output_dir / "_runs" / relative_stem
        run_output_dir.mkdir(parents=True, exist_ok=True)

        command_text = settings.mineru_command_template.format(
            input=shlex.quote(str(source_path)),
            output=shlex.quote(str(run_output_dir)),
        )
        command_args = shlex.split(command_text)
        if not command_args:
            return None

        try:
            subprocess.run(
                command_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=settings.mineru_timeout_seconds,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None

        markdown_candidates = [path for path in run_output_dir.rglob("*.md") if path.is_file()]
        if not markdown_candidates:
            return None

        selected_markdown = max(markdown_candidates, key=lambda item: item.stat().st_size)
        target_markdown = self._output_dir / relative_stem.with_suffix(".md")
        target_markdown.parent.mkdir(parents=True, exist_ok=True)
        target_markdown.write_text(selected_markdown.read_text(encoding="utf-8"), encoding="utf-8")

        meta_path = target_markdown.with_suffix(f"{target_markdown.suffix}.meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "source_path": self._relative_path(source_path),
                    "source_type": "pdf",
                    "generated_at": time.time(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return target_markdown

    def _read_manifest(self) -> dict[str, dict[str, str | float]]:
        if not self._manifest_path.exists():
            return {}
        try:
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): value
            for key, value in payload.items()
            if isinstance(key, str) and isinstance(value, dict)
        }

    def _relative_path(self, path: Path) -> str:
        if self.base_dir is None:
            return str(path)
        return str(path.relative_to(self.base_dir)).replace("\\", "/")


mineru_preprocessor = MineruPreprocessor()
