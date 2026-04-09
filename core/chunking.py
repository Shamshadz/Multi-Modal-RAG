from typing import List, Dict, Any
import re


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] = None
    ):
        """
        chunk_size: max characters per chunk
        chunk_overlap: overlap between chunks
        separators: priority-based splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    # -----------------------------
    # PUBLIC METHOD
    # -----------------------------
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None):
        """
        Returns list of chunks with metadata
        """
        splits = self._recursive_split(text)

        chunks = []
        current_chunk = ""

        for piece in splits:
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk:
                    chunks.append(self._build_chunk(current_chunk, metadata))

                current_chunk = piece

        if current_chunk:
            chunks.append(self._build_chunk(current_chunk, metadata))

        # Apply overlap
        return self._apply_overlap(chunks, metadata)

    # -----------------------------
    # RECURSIVE SPLITTING
    # -----------------------------
    def _recursive_split(self, text: str, level: int = 0) -> List[str]:
        if level >= len(self.separators):
            return [text]

        sep = self.separators[level]
        pieces = text.split(sep)

        result = []
        for piece in pieces:
            if len(piece) > self.chunk_size:
                result.extend(self._recursive_split(piece, level + 1))
            else:
                result.append(piece + sep)

        return result

    # -----------------------------
    # BUILD CHUNK OBJECT
    # -----------------------------
    def _build_chunk(self, text: str, metadata: Dict[str, Any]):
        return {
            "text": text.strip(),
            "metadata": metadata or {}
        }

    # -----------------------------
    # APPLY OVERLAP
    # -----------------------------
    def _apply_overlap(self, chunks: List[Dict], metadata: Dict[str, Any]):
        if not chunks or self.chunk_overlap <= 0:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            text = chunk["text"]

            if i > 0:
                prev_text = chunks[i - 1]["text"]
                overlap_text = prev_text[-self.chunk_overlap:]
                text = overlap_text + " " + text

            overlapped_chunks.append({
                "text": text.strip(),
                "metadata": chunk["metadata"]
            })

        return overlapped_chunks