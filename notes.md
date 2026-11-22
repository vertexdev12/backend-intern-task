# Notes

- Design: the store is a dict `doc_id -> list[Chunk]` created inside `create_app()` (no module-level state), so re-ingesting a doc id simply replaces its chunks and delete is O(1).
- `/ask` scores every chunk with a dot product (vectors are unit length), sorts by score desc with chunk id as a deterministic tie-breaker, and takes the top_k.
- Validation is done with pydantic (empty id, blank text/question, top_k 1..10, empty documents list all give 422).
- With more time: add tests for edge cases (duplicate ids in one request, ties, overlap/size errors), use numpy/vector index for large stores, add a lock for concurrent writes.
- AI tools used: Claude (for drafting and reviewing the code); I read and tested everything myself.
