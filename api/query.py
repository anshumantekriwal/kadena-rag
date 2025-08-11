from fastapi import APIRouter, HTTPException
from models.schema import QueryRequest, QueryResponse, SourceChunk
from services.chroma_service import ChromaService
from services.llm_service import LLMService
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest) -> QueryResponse:
    logger.info("Q: %s", request.query)
    try:
        chroma = ChromaService()
        retrieved, _ = chroma.search(query=request.query, top_k=request.top_k)
        logger.info("Retrieved %d chunks", len(retrieved))

        llm = LLMService()
        answer = llm.generate_answer(request.query, retrieved)

        source_chunks = [
            SourceChunk(
                id=c["id"],
                text=c["text"],
                title=c.get("title",""),
                collection=c.get("collection",""),
                source_file=c.get("source_file",""),
                section=c.get("section",""),
                links=json.loads(c.get("links", "[]")),
                score=c.get("score"),
                metadata=c.get("metadata",{})
            )
            for c in retrieved
        ]
        return QueryResponse(answer=answer, source_chunks=source_chunks)

    except Exception as e:
        logger.exception("Query error")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")