from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import query
from dotenv import load_dotenv
import os

app = FastAPI(
    title="Kadena RAG",
    description="RAG System for Kadena Documentation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query.router, tags=["Query"])

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Kadena RAG system is running"}

if __name__ == "__main__":
    import uvicorn
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000) 