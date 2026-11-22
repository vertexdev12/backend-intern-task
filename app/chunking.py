def chunk_text(text: str, size: int = 50, overlap: int = 10) -> list[str]:
    """Split `text` into overlapping chunks of words.

    - Words are separated by whitespace (str.split()).
    - Each chunk has at most `size` words, joined with a single space.
    - Consecutive chunks share exactly `overlap` words.
    - Stop when a chunk reaches the last word (no extra trailing chunk).
    - Empty / whitespace-only text returns [].
    - ValueError if size <= 0, overlap < 0 or overlap >= size.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("require size > 0, overlap >= 0 and overlap < size")

    words = text.split()
    if not words:
        return []

    step = size - overlap
    chunks: list[str] = []
    start = 0
    while True:
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks
