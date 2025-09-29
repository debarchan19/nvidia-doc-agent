from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

from rag.config import Config


class IngestPipeline:
    """Functions for indexing NVIDIA markdown docs into a Chroma vector store."""

    @classmethod
    def run(
        cls,
        *,
        config: Config,
        embeddings: Embeddings,
        persist_directory: Path | None = None,
    ) -> Chroma:
        logging.debug("Starting ingestion process")
        try:
            documents = cls._chunk_documents(
                config=config,
                documents=cls._load_documents(config.markdown_files, config),
            )
            persist_path = cls._resolve_persist_dir(config, persist_directory)
            settings = config.chroma_settings(persist_path)
            logging.info(f"Adding {len(documents)} chunks to Chroma vector store")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=str(persist_path),
                collection_name=config.chroma_collection,
                client_settings=settings,
            )
            logging.info("Successfully added documents to vector store")
            return vector_store
        except Exception as e:
            logging.error(f"Error during ingestion: {e}", exc_info=True)
            raise

    @staticmethod
    def _load_documents(files: Sequence[Path], config: Config) -> Iterable[Document]:
        logging.info(f"Starting to process {len(files)} files")
        for file_path in files:
            logging.info(f"Processing file: {file_path}")
            text = file_path.read_text(encoding="utf-8")
            metadata = {
                "source": str(file_path.relative_to(config.docs_root)),
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "title": _extract_title(text),
            }
            yield Document(page_content=text, metadata=metadata)

    @staticmethod
    def _chunk_documents(*, config: Config, documents: Iterable[Document]) -> List[Document]:
        logging.info("Starting document chunking process")
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )
            chunked: List[Document] = []
            doc_count = 0
            for doc in documents:
                doc_count += 1
                logging.info(f"Chunking document {doc_count}: {doc.metadata.get('source', 'unknown')}")
                chunked.extend(splitter.split_documents([doc]))
            return chunked
        except Exception as e:
            logging.error(f"Error during document chunking: {e}", exc_info=True)
            raise

    @staticmethod
    def _resolve_persist_dir(config: Config, override: Path | None) -> Path:
        if override is not None:
            return Path(override).expanduser().resolve()
        return config.vector_store_path()


def _extract_title(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            if stripped.startswith("#"):
                return stripped.lstrip("# ") or None
            return stripped
    return None
