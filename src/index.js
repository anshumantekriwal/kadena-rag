import express from 'express';
import bodyParser from 'body-parser';
import { fileURLToPath } from "url";
import scrapeKadenaDocs from "./scraper.js";
import Vectorizer from "./vectorizer.js";
import dotenv from "dotenv";
import { OpenAI } from "openai";

dotenv.config();

// Initialize Express
const app = express();
app.use(bodyParser.json());

class KadenaRAG {
  constructor() {
    this.vectorizer = new Vectorizer();
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  }

  async initialize() {
    try {
      console.log("Initializing Kadena RAG system...");

      // Initialize vectorizer
      await this.vectorizer.initialize();

      // // Check if we need to scrape and process docs
      // console.log("Scraping Kadena documentation...");
      // const documents = await scrapeKadenaDocs();

      console.log("Processing documents...");
      await this.vectorizer.processDocuments();

      console.log("Initialization complete!");
    } catch (error) {
      console.error("Error initializing RAG system:", error);
      throw error;
    }
  }

  async query(question, numResults = 5) {
    try {
      // Get relevant documents
      const results = await this.vectorizer.query(question, numResults);

      // Prepare context from retrieved documents
      const context = results.documents[0].join("\n\n");


      // Generate response using OpenAI
      const completion = await this.openai.chat.completions.create({
        model: "o3-mini",
        messages: [
          {
            role: "system",
            content:
              "You are K-Agent: An AI agent that answers questions about Kadena blockchain. " +
              "Use the provided context to answer questions accurately. " +
              "Be detailed in your answers " +
              "If you're not sure about something, say so." +
              "Do NOT use Markdown formatting in your responses.",
          },
          {
            role: "user",
            content: `Context: ${context}\n\nQuestion: ${question}`,
          },
        ],
      });

      return {
        answer: completion.choices[0].message.content,
        sources: results.metadatas[0].map((meta) => ({
          title: meta.title,
          source: meta.source,
        })),
      };
    } catch (error) {
      console.error("Error querying RAG system:", error);
      throw error;
    }
  }
}

// Initialize and start server
async function startServer() {
  try {
    const rag = new KadenaRAG();
    await rag.initialize();
    
    // Import routes after initialization
    const router = (await import('./api/rag.js')).default(rag);
    
    app.use('/api', router);
    
    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
      console.log(`RAG endpoint available at http://localhost:${PORT}/api/rag`);
    });
  } catch (error) {
    console.error("Failed to start server:", error);
    process.exit(1);
  }
}

// Start the server if run directly
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  startServer();
}

export { KadenaRAG, app };
