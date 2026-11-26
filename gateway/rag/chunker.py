import re

from gateway.rag.models import Document, DocumentChunk

class TextChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        text = document.text.strip()

        if not text:
            return []

        segments = self._split_into_segments(text)
        chunk_texts = self._build_chunks(segments)

        return [
            DocumentChunk(
                chunk_id=f"{document.document_id}-{position}",
                document_id=document.document_id,
                text=chunk_text,
                position=position,
                metadata=document.metadata.copy(),
            )
            for position, chunk_text in enumerate(chunk_texts)
        ]

    @staticmethod
    def _split_into_segments(text: str) -> list[str]:
        paragraphs = re.split(r"\n\s*\n+", text)
        segments: list[str] = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            sentences = re.split(
                r"(?<=[.!?])\s+",
                paragraph,
            )

            segments.extend(
                sentence.strip()
                for sentence in sentences
                if sentence.strip()
            )

        return segments

    def _build_chunks(self, segments: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""

        for segment in segments:
            candidate = (
                f"{current} {segment}".strip()
                if current
                else segment
            )

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)

            current = self._overlap_text(current)

            candidate = (
                f"{current} {segment}".strip()
                if current
                else segment
            )

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            hard_chunks = self._split_large_segment(segment)

            if current:
                chunks.append(current)
                current = ""

            chunks.extend(hard_chunks[:-1])
            current = hard_chunks[-1]

        if current:
            chunks.append(current)

        return chunks

    def _overlap_text(self, text: str) -> str:
        if not text or self.overlap == 0:
            return ""

        return text[-self.overlap:].lstrip()

    def _split_large_segment(self, segment: str) -> list[str]:
        chunks: list[str] = []
        start = 0

        while start < len(segment):
            end = min(start + self.chunk_size, len(segment))
            chunks.append(segment[start:end])

            if end == len(segment):
                break

            start = end - self.overlap

        return chunks