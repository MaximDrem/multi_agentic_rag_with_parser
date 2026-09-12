import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from common.chunk_store import ChunkDatabase
from indexing.chunker import chunk_directory
from indexing.faiss_indexer import FaissIndexer
from indexing.bm25_indexer import BM25Indexer


load_dotenv()


def main():
    markdown_dir = Path(os.getenv("MARKDOWN_DIR", "./var/markdown_output_clean"))
    chunks_path = Path(os.getenv("CHUNKS_PATH", "./var/data/chunks.pkl"))
    faiss_path = Path(os.getenv("FAISS_INDEX_PATH", "./var/data/faiss_index"))
    if faiss_path.is_dir() or str(faiss_path).endswith("/"):
        faiss_path = faiss_path / "index.faiss"
    bm25_path = Path(os.getenv("BM25_INDEX_PATH", "./var/data/bm25_index.pkl"))

    print(f"Chunking markdown files from {markdown_dir} ...")
    chunks = chunk_directory(markdown_dir)
    print(f"Total chunks: {len(chunks)}")

    db = ChunkDatabase(chunks)
    db.save(chunks_path)
    print(f"Saved chunks to {chunks_path}")

    print("Building FAISS index ...")
    faiss_indexer = FaissIndexer()
    faiss_indexer.load_model()
    faiss_indexer.build(chunks)
    faiss_indexer.save(faiss_path)
    print(f"Saved FAISS index to {faiss_path}")

    print("Building BM25 index ...")
    bm25_indexer = BM25Indexer()
    bm25_indexer.build(chunks)
    bm25_indexer.save(bm25_path)
    print(f"Saved BM25 index to {bm25_path}")


if __name__ == "__main__":
    main()
