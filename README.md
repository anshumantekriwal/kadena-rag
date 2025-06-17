# Kadena RAG

A Retrieval-Augmented Generation (RAG) system for Kadena documentation. This system uses ChromaDB for vector storage and OpenAI's GPT-4 for generating answers to questions about Kadena.

## Features

- Vector storage of Kadena documentation using ChromaDB
- Semantic search using OpenAI embeddings
- Context-aware question answering using GPT-4
- FastAPI backend with RESTful API
- Modular architecture with separate services for different components

## Project Structure

```
.
├── api/            # API routes and endpoints
├── data/           # Documentation data and storage
├── models/         # Data models and schemas
├── services/       # Core services (ChromaDB, LLM)
├── main.py         # Application entry point
├── requirements.txt # Project dependencies
└── README.md       # This file
```

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd kadena-rag
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:

```bash
python main.py
```

The server will start on `http://localhost:8000`

## Dependencies

The project uses the following main dependencies:

- FastAPI (v0.109.2+) - Web framework
- LangChain (v0.1.9+) - LLM framework
- ChromaDB (v0.4.22+) - Vector database
- OpenAI (v1.12.0+) - LLM and embeddings
- Pydantic (v2.6.1+) - Data validation

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

   - Stores document embeddings
   - Performs semantic search
   - Manages document persistence

2. **LLM Service**: Formats retrieved chunks and generates answers using GPT-4

   - Processes user queries
   - Formats context for the LLM
   - Generates coherent responses

3. **FastAPI Backend**: Provides the REST API interface
   - Handles HTTP requests
   - Validates input/output
   - Manages error responses

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

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT
