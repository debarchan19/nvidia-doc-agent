from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Iterator, Tuple

from chromadb.config import Settings


@dataclass(slots=True)
class Config:
    """Application configuration for working with NVIDIA markdown docs."""

    docs_root: Path
    markdown_suffixes: Tuple[str, ...] = (".md",)
    chunk_size: int = 1200
    chunk_overlap: int = 200
    vector_store_dir: Path | None = None
    chroma_collection: str = "nvidia_docs"
    chroma_anonymized_telemetry: bool = False
    modernbert_model_path: str | None = None  # Path to local ModernBERT model
    _cached_files: Tuple[Path, ...] | None = field(default=None, init=False, repr=False)

    ENV_DOCS_ROOT = "NVIDIA_DOCS_ROOT"
    ENV_VECTOR_STORE_DIR = "NVIDIA_VECTOR_STORE_DIR"
    ENV_MODERNBERT_PATH = "NVIDIA_MODERNBERT_PATH"

    def __post_init__(self) -> None:
        self.docs_root = Path(self.docs_root).expanduser().resolve()
        if self.vector_store_dir is not None:
            self.vector_store_dir = Path(self.vector_store_dir).expanduser().resolve()

    @classmethod
    def from_env(cls) -> "Config":
        env_path = os.getenv(cls.ENV_DOCS_ROOT)
        env_vector_dir = os.getenv(cls.ENV_VECTOR_STORE_DIR)
        if env_path:
            return cls(
                docs_root=Path(env_path),
                vector_store_dir=Path(env_vector_dir) if env_vector_dir else None,
            )
        return cls.from_project_root(
            vector_store_dir=Path(env_vector_dir) if env_vector_dir else None
        )

    @classmethod
    def from_project_root(cls, *, vector_store_dir: Path | None = None) -> "Config":
        project_root = Path(__file__).resolve().parent.parent
        default_root = project_root / "nvidia_docs_md"
        default_vector_dir = vector_store_dir or project_root / "rag" / "vector_store"
        return cls(docs_root=default_root, vector_store_dir=default_vector_dir)

    @property
    def markdown_files(self) -> Tuple[Path, ...]:
        if self._cached_files is None:
            self._cached_files = tuple(self.iter_markdown_files())
        return self._cached_files

    def iter_markdown_files(self) -> Iterator[Path]:
        for candidate in self.docs_root.rglob("*"):
            if (
                candidate.is_file()
                and candidate.suffix.lower() in self.markdown_suffixes
            ):
                yield candidate

    def refresh_markdown_cache(self) -> None:
        self._cached_files = None

    def count_markdown_files(self) -> int:
        return len(self.markdown_files)

    def vector_store_path(self) -> Path:
        target = (
            self.vector_store_dir or (self.docs_root.parent / "rag" / "vector_store")
        ).resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    def chroma_settings(self, persist_directory: Path) -> Settings:
        return Settings(
            anonymized_telemetry=False,
            persist_directory=str(persist_directory),
            is_persistent=True,
        )


config = Config.from_env()
