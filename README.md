# Kadena RAG

A Retrieval-Augmented Generation (RAG) system for Kadena documentation. This system uses ChromaDB for vector storage and OpenAI's GPT-4 for generating answers to questions about Kadena.

## Features

- Vector storage of Kadena documentation using ChromaDB
- Semantic search using OpenAI embeddings
- Context-aware question answering using GPT-4
- FastAPI backend with RESTful API

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

4. Run the application:

```bash
python main.py
```

The server will start on `http://localhost:8000`

## API Usage

### Query Endpoint

Send a POST request to `/api/v1/query` with the following JSON body:

```json
{
  "query": "What is Kadena's mining algorithm?"
}
```

The response will include:

- `answer`: The generated answer to your question
- `source_chunks`: The relevant chunks of text used to generate the answer

Example response:

```json
{
  "answer": "Kadena uses the Blake2s mining algorithm, which is ASIC-resistant and designed to be more energy-efficient than Bitcoin's SHA-256. The mining difficulty adjusts automatically to maintain a consistent block time across all chains.",
  "source_chunks": [
    "Kadena's mining algorithm is Blake2s, which is ASIC-resistant and designed to be more energy-efficient than Bitcoin's SHA-256. The mining difficulty adjusts automatically to maintain a consistent block time across all chains."
  ]
}
```

## Architecture

The system consists of three main components:

1. **ChromaDB Service**: Handles vector storage and retrieval of document chunks
2. **LLM Service**: Formats retrieved chunks and generates answers using GPT-4
3. **FastAPI Backend**: Provides the REST API interface

## Data Structure

The documentation is stored in `data/docs.json` in the following format:

```json
[
  {
    "content": "Document content here",
    "source": "Source of the document"
  }
]
```

## License

MIT
