import { ChromaClient, OpenAIEmbeddingFunction } from "chromadb";
import dotenv from "dotenv";
import path from "path";
import fs from "fs-extra";

dotenv.config();

if (!process.env.OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY is required in .env file");
}

const COLLECTION_NAME = "kadena_docs";
const CHUNK_SIZE = 500; // characters per chunk

class Vectorizer {
  constructor() {
    this.client = new ChromaClient();
    this.embeddingFunction = new OpenAIEmbeddingFunction(
      process.env.OPENAI_API_KEY
    );
  }

  async initialize() {
    try {
      // Try to get existing collection first
      try {
        this.collection = await this.client.getCollection({
          name: COLLECTION_NAME,
          embeddingFunction: this.embeddingFunction,
        });
        console.log("Using existing collection:", COLLECTION_NAME);
      } catch (error) {
        // If collection doesn't exist, create it
        this.collection = await this.client.createCollection({
          name: COLLECTION_NAME,
          embeddingFunction: this.embeddingFunction,
        });
        console.log("Created new collection:", COLLECTION_NAME);
      }
    } catch (error) {
      console.error("Error initializing vectorizer:", error);
      throw error;
    }
  }

  splitIntoChunks(text, metadata) {
    const chunks = [];
    const words = text.split(" ");
    let currentChunk = "";

    for (const word of words) {
      if ((currentChunk + " " + word).length <= CHUNK_SIZE) {
        currentChunk += (currentChunk ? " " : "") + word;
      } else {
        if (currentChunk) {
          chunks.push({
            text: currentChunk,
            metadata: { ...metadata },
          });
        }
        currentChunk = word;
      }
    }

    if (currentChunk) {
      chunks.push({
        text: currentChunk,
        metadata: { ...metadata },
      });
    }

    return chunks;
  }

  async processDocuments() {
    try {
      // Check if embeddings already exist by issuing a query with an empty string
      const check = await this.collection.query({
        queryTexts: [" "],
        nResults: 1,
      });
      if (check && check.ids && check.ids.length > 0) {
        console.log("Embeddings already exist, skipping processing.");
        return;
      }

      const docsPath = path.join(process.cwd(), "data", "processed_docs.json");
      const documents = await fs.readJSON(docsPath);

      let allChunks = [];
      let ids = [];
      let texts = [];
      let metadatas = [];

      documents.forEach((doc, docIndex) => {
        const chunks = this.splitIntoChunks(doc.content, {
          source: doc.source,
          title: doc.title,
        });

        chunks.forEach((chunk, chunkIndex) => {
          const id = `${docIndex}-${chunkIndex}`;
          ids.push(id);
          texts.push(chunk.text);
          metadatas.push(chunk.metadata);
        });

        allChunks.push(...chunks);
      });

      // Add documents to collection
      await this.collection.add({
        ids: ids,
        documents: texts,
        metadatas: metadatas,
      });

      console.log(
        `Processed ${allChunks.length} chunks from ${documents.length} documents`
      );
    } catch (error) {
      console.error("Error processing documents:", error);
      throw error;
    }
  }

  async query(queryText, numResults = 5) {
    try {
      const results = await this.collection.query({
        queryTexts: [queryText],
        nResults: numResults,
      });

      return results;
    } catch (error) {
      console.error("Error querying collection:", error);
      throw error;
    }
  }
}

export default Vectorizer;
