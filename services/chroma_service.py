from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaService:
    def __init__(self):
        load_dotenv()
        self.persist_directory = "./chromadb"
        self.embedding_function = OpenAIEmbeddings(model='text-embedding-3-small')
        self.collection_name = "kadena-docs"
        logger.info("Initializing ChromaService with collection: %s", self.collection_name)
        self._initialize_collection()

    def _initialize_collection(self) -> None:
        """Initialize the collection with Kadena docs"""
        # Check if collection already exists
        if os.path.exists(self.persist_directory):
            logger.info("Collection already exists at %s", self.persist_directory)
            return

        # Load docs from JSON
        with open("data/docs.json", "r") as f:
            docs_data = json.load(f)
        logger.info("Loaded %d documents from data/docs.json", len(docs_data))

        # Convert to Document objects
        documents = []
        for idx, doc in enumerate(docs_data):
            documents.append(Document(
                page_content=doc["content"],
                metadata={
                    "source": doc.get("source", "unknown"),
                    "chunk_index": idx
                }
            ))
        logger.info("Converted %d documents to Document objects", len(documents))

        # Create collection and add documents
        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )
        vector_store.add_documents(documents)
        logger.info("Added %d documents to ChromaDB collection: %s", len(documents), self.collection_name)

    def search(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "mmr"
    ) -> Tuple[List[str], List[dict]]:
        """
        Search for relevant chunks using MMR or similarity search
        
        Args:
            query: Search query
            top_k: Number of results to return
            search_type: "mmr" or "similarity"
            
        Returns:
            Tuple of (chunks, metadata)
        """
        logger.info("Searching for query: %s with top_k=%d and search_type=%s", query, top_k, search_type)
        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )
        
        search_kwargs = {
            "k": top_k,
        }
        
        if search_type == "mmr":
            search_kwargs.update({
                "fetch_k": min(10, top_k * 2),
                "lambda_mult": 0.25
            })
            
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        # Get results
        results = retriever.invoke(query)
        logger.info("Retrieved %d results for query: %s", len(results), query)
        
        # Extract chunks and metadata
        chunks = [doc.page_content for doc in results]
        metadata = [doc.metadata for doc in results]
        
        return chunks, metadata 