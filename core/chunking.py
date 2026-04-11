"""
chunking.py
===========
Responsible for splitting raw text into overlapping chunks suitable for embedding
and semantic retrieval.

Design decisions
----------------
* Recursive splitting: tries broad separators first (paragraphs, newlines, sentences)
  before falling back to word or character splits, preserving semantic boundaries.
* Sliding-window overlap: each chunk prepends the tail of the previous chunk so
  that context spanning a boundary is never lost at query time.
* Metadata passthrough: every chunk inherits the caller-supplied metadata dict,
  making it trivial to trace a chunk back to its source document.

Typical usage
-------------
>>> chunker = TextChunker(chunk_size=500, chunk_overlap=100)
>>> chunks = chunker.chunk_text(raw_text, metadata={"source": "report.txt"})
>>> for c in chunks:
...     print(c["text"][:80], c["metadata"])
"""

from typing import List, Dict, Any


class TextChunker:
    """
    Splits a long string into overlapping text chunks using recursive separators.

    Parameters
    ----------
    chunk_size : int
        Maximum number of characters per chunk (before overlap is added).
    chunk_overlap : int
        Number of characters taken from the end of the previous chunk and
        prepended to the current chunk.
    separators : list[str] or None
        Priority-ordered list of separators used for splitting.
        Defaults to ``["\n\n", "\n", ". ", " "]``.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Split *text* into overlapping chunks.

        Parameters
        ----------
        text : str
            The full document text to split.
        metadata : dict, optional
            Arbitrary key/value pairs attached to every produced chunk
            (e.g. ``{"source": "annual_report.pdf", "page": 3}``).

        Returns
        -------
        list[dict]
            Each element has the shape::

                {
                    "text": "<chunk text>",
                    "metadata": { ...caller metadata... }
                }
        """
        if not text or not text.strip():
            return []

        splits = self._recursive_split(text)
        raw_chunks: List[Dict[str, Any]] = []
        current_chunk = ""

        for piece in splits:
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk.strip():
                    raw_chunks.append(self._build_chunk(current_chunk, metadata))
                current_chunk = piece

        if current_chunk.strip():
            raw_chunks.append(self._build_chunk(current_chunk, metadata))

        return self._apply_overlap(raw_chunks, metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str, level: int = 0) -> List[str]:
        """
        Recursively split *text* using ``self.separators[level]``.

        If a piece is still larger than ``chunk_size`` after splitting at this
        level, the next separator is tried on that piece.
        """
        if level >= len(self.separators):
            # Hard fallback: character-level slicing
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = self.separators[level]
        pieces = text.split(sep)

        result: List[str] = []
        for piece in pieces:
            if not piece:
                continue
            if len(piece) > self.chunk_size:
                result.extend(self._recursive_split(piece, level + 1))
            else:
                # Re-attach separator so context is preserved
                result.append(piece + sep)

        return result

    def _build_chunk(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap stripped text and metadata into the standard chunk dict."""
        return {
            "text": text.strip(),
            "metadata": metadata.copy() if metadata else {},
        }

    def _apply_overlap(
        self,
        chunks: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Prepend the last ``chunk_overlap`` characters of chunk *i-1* to chunk
        *i*, so that sentences crossing a boundary appear in both chunks.
        """
        if not chunks or self.chunk_overlap <= 0:
            return chunks

        overlapped: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            if i > 0:
                prev_tail = chunks[i - 1]["text"][-self.chunk_overlap :]
                text = prev_tail + " " + text

            overlapped.append({
                "text": text.strip(),
                "metadata": chunk["metadata"],
            })

        return overlapped