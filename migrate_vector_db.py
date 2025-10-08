#!/usr/bin/env python3
"""
Vector Database Migration Script
Handles cross-platform migration of ChromaDB vector stores between Linux servers and Mac clients.
"""

import argparse
import logging
import shutil
import tarfile
import time
from pathlib import Path
from typing import Optional

def setup_logging():
    """Configure logging for migration operations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("migration.log", mode="a"),
        ],
    )

def create_vector_db_package(
    vector_store_path: Path,
    output_path: Path,
    include_metadata: bool = True
) -> Path:
    """
    Package vector database for migration.
    
    Args:
        vector_store_path: Path to the ChromaDB vector store
        output_path: Output directory for the package
        include_metadata: Whether to include metadata files
    
    Returns:
        Path to the created package
    """
    logger = logging.getLogger(__name__)
    
    if not vector_store_path.exists():
        raise FileNotFoundError(f"Vector store not found: {vector_store_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate package name with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    package_name = f"nvidia_vector_db_{timestamp}.tar.gz"
    package_path = output_path / package_name
    
    logger.info(f"Creating vector database package: {package_path}")
    logger.info(f"Source: {vector_store_path}")
    
    with tarfile.open(package_path, "w:gz") as tar:
        # Add vector store directory
        tar.add(vector_store_path, arcname="vector_store")
        
        if include_metadata:
            # Add configuration and metadata
            project_root = Path(__file__).parent
            metadata_files = [
                "rag/config.py",
                "requirements.txt",
                "pyproject.toml"
            ]
            
            for file_path in metadata_files:
                full_path = project_root / file_path
                if full_path.exists():
                    tar.add(full_path, arcname=f"metadata/{file_path}")
                    logger.info(f"Added metadata: {file_path}")
    
    file_size = package_path.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"✅ Package created successfully!")
    logger.info(f"   📦 File: {package_path}")
    logger.info(f"   📏 Size: {file_size:.2f} MB")
    
    return package_path

def extract_vector_db_package(
    package_path: Path,
    target_path: Path,
    overwrite: bool = False
) -> Path:
    """
    Extract vector database package.
    
    Args:
        package_path: Path to the package file
        target_path: Target directory for extraction
        overwrite: Whether to overwrite existing files
    
    Returns:
        Path to the extracted vector store
    """
    logger = logging.getLogger(__name__)
    
    if not package_path.exists():
        raise FileNotFoundError(f"Package not found: {package_path}")
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    vector_store_target = target_path / "vector_store"
    
    if vector_store_target.exists() and not overwrite:
        raise FileExistsError(
            f"Vector store already exists: {vector_store_target}. "
            "Use --overwrite to replace it."
        )
    
    logger.info(f"Extracting vector database package: {package_path}")
    logger.info(f"Target: {target_path}")
    
    with tarfile.open(package_path, "r:gz") as tar:
        tar.extractall(target_path)
    
    logger.info("✅ Package extracted successfully!")
    logger.info(f"   📁 Vector store: {vector_store_target}")
    
    return vector_store_target

def validate_vector_db(vector_store_path: Path) -> bool:
    """
    Validate vector database integrity.
    
    Args:
        vector_store_path: Path to the vector store
    
    Returns:
        True if valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check required files
        required_files = ["chroma.sqlite3"]
        for file_name in required_files:
            file_path = vector_store_path / file_name
            if not file_path.exists():
                logger.error(f"Missing required file: {file_name}")
                return False
        
        # Try to load with ChromaDB
        import chromadb
        from chromadb.config import Settings
        
        settings = Settings(
            persist_directory=str(vector_store_path),
            anonymized_telemetry=False,
            is_persistent=True,
        )
        
        client = chromadb.PersistentClient(settings=settings)
        collections = client.list_collections()
        
        logger.info(f"✅ Vector database validation successful!")
        logger.info(f"   📊 Collections found: {len(collections)}")
        
        for collection in collections:
            count = collection.count()
            logger.info(f"   📚 Collection '{collection.name}': {count} documents")
        
        return True
        
    except Exception as e:
        logger.error(f"Vector database validation failed: {e}")
        return False

def create_migration_info(vector_store_path: Path) -> dict:
    """Create migration information file."""
    import platform
    import sys
    from datetime import datetime
    
    try:
        from rag.config import config
        docs_count = config.count_markdown_files()
    except:
        docs_count = "unknown"
    
    info = {
        "migration_timestamp": datetime.now().isoformat(),
        "source_platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": sys.version,
        },
        "vector_store_info": {
            "path": str(vector_store_path),
            "size_mb": sum(f.stat().st_size for f in vector_store_path.rglob("*") if f.is_file()) / (1024 * 1024),
            "documents_processed": docs_count,
        },
        "embeddings_info": {
            "model": "ModernBERT-base",
            "dimension": 768,
            "chunk_size": 1200,
            "chunk_overlap": 200,
        }
    }
    
    return info

def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="NVIDIA Docs Vector Database Migration")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Package command
    package_parser = subparsers.add_parser("package", help="Package vector database for migration")
    package_parser.add_argument("--vector-store", type=Path, required=True, help="Path to vector store")
    package_parser.add_argument("--output", type=Path, default=Path("."), help="Output directory")
    package_parser.add_argument("--no-metadata", action="store_true", help="Skip metadata files")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract vector database package")
    extract_parser.add_argument("--package", type=Path, required=True, help="Path to package file")
    extract_parser.add_argument("--target", type=Path, default=Path("."), help="Target directory")
    extract_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate vector database")
    validate_parser.add_argument("--vector-store", type=Path, required=True, help="Path to vector store")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if args.command == "package":
        try:
            package_path = create_vector_db_package(
                args.vector_store,
                args.output,
                include_metadata=not args.no_metadata
            )
            
            # Create migration info
            info = create_migration_info(args.vector_store)
            info_path = args.output / f"migration_info_{time.strftime('%Y%m%d_%H%M%S')}.json"
            
            import json
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            
            logger.info(f"📋 Migration info saved: {info_path}")
            
        except Exception as e:
            logger.error(f"Packaging failed: {e}")
            return 1
    
    elif args.command == "extract":
        try:
            vector_store_path = extract_vector_db_package(
                args.package,
                args.target,
                args.overwrite
            )
            
            # Validate after extraction
            if validate_vector_db(vector_store_path):
                logger.info("🎉 Migration completed successfully!")
            else:
                logger.error("⚠️  Migration completed but validation failed")
                return 1
                
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return 1
    
    elif args.command == "validate":
        try:
            if validate_vector_db(args.vector_store):
                logger.info("✅ Vector database is valid!")
                return 0
            else:
                logger.error("❌ Vector database validation failed!")
                return 1
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())