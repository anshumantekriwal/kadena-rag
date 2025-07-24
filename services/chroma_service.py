from typing import List, Tuple, Dict, Any
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
        
        # Define separate collections
        self.collections = {
            "documentation": {
                "name": "kadena-docs",
                "sources": [{"file": "data/docs.json", "type": "documentation"}]
            },
            "ecosystem": {
                "name": "kadena-ecosystem", 
                "sources": [{"file": "data/ecosystem-projects.json", "type": "ecosystem"}]
            },
            "info": {
                "name": "kadena-info",
                "sources": [
                    {"file": "data/kadena-info.json", "type": "kadena-info"},
                    {"file": "data/miscellaneous.json", "type": "miscellaneous"}
                ]
            }
        }
        
        logger.info("Initializing ChromaService with separate collections")
        self._initialize_collections()

    def _initialize_collections(self) -> None:
        """Initialize separate collections for each data source group"""
        # Check if collections already exist
        if os.path.exists(self.persist_directory):
            logger.info("Collections already exist at %s", self.persist_directory)
            return

        # Create each collection separately
        for collection_key, collection_config in self.collections.items():
            collection_name = collection_config["name"]
            data_sources = collection_config["sources"]
            
            logger.info("Creating collection: %s", collection_name)
            
            # Load and combine documents for this collection
            collection_documents = []
            total_docs_loaded = 0
            
            for data_source in data_sources:
                file_path = data_source["file"]
                data_type = data_source["type"]
                
                try:
                    with open(file_path, "r", encoding='utf-8') as f:
                        docs_data = json.load(f)
                    logger.info("Loaded %d documents from %s for collection %s", 
                               len(docs_data), file_path, collection_name)
                    
                    # Convert to Document objects with comprehensive metadata
                    for idx, doc in enumerate(docs_data):
                        # Create rich metadata
                        metadata = {
                            "source": doc.get("source", "unknown"),
                            "data_type": data_type,
                            "data_file": file_path,
                            "title": doc.get("title", ""),
                            "chunk_index": total_docs_loaded + idx,
                            "collection": collection_key
                        }
                        
                        # Add content length for potential filtering/ranking
                        metadata["content_length"] = len(doc["content"])
                        
                        # Add specific metadata based on data type
                        if data_type == "documentation":
                            # For docs.json, source contains the actual file path
                            metadata["doc_category"] = self._categorize_doc_source(doc.get("source", ""))
                        elif data_type == "ecosystem":
                            # Extract project category from title if available
                            title = doc.get("title", "")
                            if " - " in title:
                                project_name = title.split(" - ")[0]
                                metadata["project_name"] = project_name
                        
                        collection_documents.append(Document(
                            page_content=doc["content"],
                            metadata=metadata
                        ))
                    
                    total_docs_loaded += len(docs_data)
                    
                except Exception as e:
                    logger.error("Error loading %s: %s", file_path, e)
                    continue
            
            logger.info("Total documents for collection %s: %d", collection_name, len(collection_documents))
            
            # Create this collection and add documents
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
            
            # Add documents in batches for better performance
            batch_size = 100
            for i in range(0, len(collection_documents), batch_size):
                batch = collection_documents[i:i + batch_size]
                vector_store.add_documents(batch)
                logger.info("Added batch %d-%d documents to %s", 
                           i, min(i + batch_size, len(collection_documents)), collection_name)
            
            logger.info("Successfully created collection %s with %d documents", 
                       collection_name, len(collection_documents))
    
    def _categorize_doc_source(self, source: str) -> str:
        """Categorize documentation source into broader categories"""
        if not source:
            return "unknown"
        
        source_lower = source.lower()
        if "pact-5" in source_lower:
            return "pact-language"
        elif "api" in source_lower:
            return "api-reference"
        elif "guides" in source_lower:
            return "guides"
        elif "reference" in source_lower:
            return "reference"
        elif "smart-contracts" in source_lower:
            return "smart-contracts"
        elif "coding-projects" in source_lower:
            return "coding-projects"
        elif "resources" in source_lower:
            return "resources"
        else:
            return "general"

    def search(
        self,
        query: str,
        top_k: int = 12,
        search_type: str = "mmr"
    ) -> Tuple[List[str], List[dict]]:
        """
        Search across separate collections, getting 4 chunks from each collection
        
        Args:
            query: Search query
            top_k: Number of results to return (default: 12 for 4 from each of 3 collections)
            search_type: Search type (fixed to "mmr")
            
        Returns:
            Tuple of (chunks, metadata)
        """
        logger.info("Searching across separate collections for query: %s with top_k=%d, search_type=%s", 
                   query, top_k, search_type)
        
        all_results = []
        chunks_per_collection = 4
        
        # Search each collection separately
        for collection_key, collection_config in self.collections.items():
            collection_name = collection_config["name"]
            
            try:
                vector_store = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embedding_function,
                    persist_directory=self.persist_directory
                )
                
                # Search configuration for this collection
                search_kwargs = {
                    "k": chunks_per_collection,
                    "fetch_k": min(10, chunks_per_collection * 2),
                    "lambda_mult": 0.25
                }
                
                retriever = vector_store.as_retriever(
                    search_type=search_type,
                    search_kwargs=search_kwargs
                )
                
                # Get results from this collection
                collection_results = retriever.invoke(query)
                logger.info("Retrieved %d results from %s collection", 
                           len(collection_results), collection_key)
                
                # Add collection identifier to metadata and extend results
                for doc in collection_results:
                    doc.metadata["search_collection"] = collection_key
                all_results.extend(collection_results)
                
            except Exception as e:
                logger.error("Error searching collection %s: %s", collection_name, e)
                continue
        
        # Trim to exact top_k if we have more results than requested
        all_results = all_results[:top_k]
        
        logger.info("Final selection: %d results total from %d collections", 
                   len(all_results), len(self.collections))
        
        # Extract chunks and metadata
        chunks = [doc.page_content for doc in all_results]
        metadata = [doc.metadata for doc in all_results]
        
        return chunks, metadata

    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all collections for debugging and monitoring
        
        Returns:
            Dictionary containing statistics for all collections
        """
        all_stats = {
            "total_collections": len(self.collections),
            "collections": {}
        }
        
        for collection_key, collection_config in self.collections.items():
            collection_name = collection_config["name"]
            
            try:
                vector_store = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embedding_function,
                    persist_directory=self.persist_directory
                )
                
                # Get sample documents from this collection
                sample_results = vector_store.similarity_search("kadena", k=50)
                
                collection_stats = {
                    "collection_name": collection_name,
                    "sample_size": len(sample_results),
                    "data_types": {},
                    "doc_categories": {},
                    "content_length_stats": {"min": float('inf'), "max": 0, "avg": 0}
                }
                
                total_length = 0
                for doc in sample_results:
                    # Count by data type
                    data_type = doc.metadata.get("data_type", "unknown")
                    collection_stats["data_types"][data_type] = collection_stats["data_types"].get(data_type, 0) + 1
                    
                    # Count by doc category (for documentation)
                    doc_category = doc.metadata.get("doc_category", "")
                    if doc_category:
                        collection_stats["doc_categories"][doc_category] = collection_stats["doc_categories"].get(doc_category, 0) + 1
                    
                    # Content length stats
                    content_length = doc.metadata.get("content_length", len(doc.page_content))
                    collection_stats["content_length_stats"]["min"] = min(collection_stats["content_length_stats"]["min"], content_length)
                    collection_stats["content_length_stats"]["max"] = max(collection_stats["content_length_stats"]["max"], content_length)
                    total_length += content_length
                
                if sample_results:
                    collection_stats["content_length_stats"]["avg"] = total_length / len(sample_results)
                
                all_stats["collections"][collection_key] = collection_stats
                
            except Exception as e:
                logger.error("Error getting stats for collection %s: %s", collection_name, e)
                all_stats["collections"][collection_key] = {"error": str(e)}
        
        return all_stats 