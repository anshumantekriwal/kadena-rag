import os
from typing import List
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = "gpt-4o-mini"
        logger.info("Initialized LLMService with model: %s", self.model)

    def format_context(self, chunks: List[str]) -> str:
        """
        Format and structure retrieved chunks
        
        Args:
            chunks: List of raw text chunks from ChromaDB
            
        Returns:
            str: Cleaned and structured context
        """
        logger.info("Formatting %d chunks into context", len(chunks))
        prompt = self._get_formatting_prompt(chunks)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                }
            ],
        )
        logger.info("Context formatting completed")
        return response.choices[0].message.content

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer based on the query and context
        
        Args:
            query: User's question
            context: Formatted context from retrieved chunks
            
        Returns:
            str: Generated answer
        """
        logger.info("Generating answer for query: %s", query)
        prompt = f'''You are a helpful assistant that answers questions about Kadena blockchain.
Use the following context to answer the question. 

Context:
{context}

Question: {query}

Answer:'''

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about Kadena blockchain."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        logger.info("Answer generation completed")
        return response.choices[0].message.content

    def _get_formatting_prompt(self, chunks: List[str]) -> str:
        """Create the formatting prompt"""
        return f'''
You are a formatting and organization assistant.

Your job is to take the raw information retrieved by a RAG system (provided below) 
and process it to create a clear, well-structured, and logically ordered context.
This context will be used by another model to answer a user query, so you must not 
answer the query yourself.

Instructions:
- Organize the information into sections or bullet points
- Remove duplicates and irrelevant or conflicting data
- Preserve technical or factual accuracy
- Do not fabricate or infer missing information
- Make the result easy for another model to read and use as direct context
- Ensure all the important information from the retrieved chunks is present

Below is the raw retrieved data:
---
{chr(10).join(chunks)}
---

Return only the cleaned and structured context below.

Context:''' 