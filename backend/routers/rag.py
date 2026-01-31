from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.rag import CodeRetriever, RAGError


router = APIRouter(tags=["rag"])

retriever: CodeRetriever | None = None


def get_retriever() -> CodeRetriever:
    """Get or create the global retriever instance."""
    global retriever
    if retriever is None:
        retriever = CodeRetriever()
    return retriever


class IndexRequest(BaseModel):
    files: dict[str, str]
    """Files to index. Format: {path: content}"""


class IndexResponse(BaseModel):
    indexed: int
    total: int


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5


class SearchResult(BaseModel):
    path: str
    content: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


@router.post("/index", response_model=IndexResponse)
async def index_files(
    request: IndexRequest,
    ret: CodeRetriever = Depends(get_retriever),
) -> IndexResponse:
    """Index files for RAG search."""
    try:
        ids = await ret.index_files(request.files)
        return IndexResponse(indexed=len(ids), total=ret.count())
    except RAGError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {e}",
        )


@router.post("/search", response_model=SearchResponse)
async def search_code(
    request: SearchRequest,
    ret: CodeRetriever = Depends(get_retriever),
) -> SearchResponse:
    """Search indexed code."""
    try:
        results = await ret.search(request.query, request.n_results)
        return SearchResponse(results=[SearchResult(**r) for r in results])
    except RAGError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e}",
        )


@router.get("/stats")
async def rag_stats(ret: CodeRetriever = Depends(get_retriever)) -> dict:
    """Get RAG index statistics."""
    return {"indexed_files": ret.count()}


@router.delete("/clear")
async def clear_index(ret: CodeRetriever = Depends(get_retriever)) -> dict:
    """Clear the RAG index."""
    ret.clear()
    return {"status": "cleared"}
