from fastapi import APIRouter, HTTPException
from models.schema import QueryRequest, QueryResponse
from services.chroma_service import ChromaService
from services.llm_service import LLMService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest) -> QueryResponse:
    """
    Process a user query with RAG
    
    This endpoint:
    1. Retrieves relevant chunks from ChromaDB
    2. Formats the chunks using LLM
    3. Generates a final answer using the formatted context
    """
    logger.info("Received query request: %s", request.query)
    try:
        # Get relevant chunks
        chroma_service = ChromaService()
        chunks, _ = chroma_service.search(
            query=request.query,
        )
        logger.info("Retrieved %d chunks from ChromaDB", len(chunks))
        
        # Format chunks and generate answer
        llm_service = LLMService()
        formatted_context = llm_service.format_context(chunks)
        answer = llm_service.generate_answer(request.query, formatted_context)
        logger.info("Generated answer for query: %s", request.query)
        
        return QueryResponse(
            answer=answer,
            source_chunks=chunks,
        )
        
    except Exception as e:
        logger.error("Error processing query: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        ) 