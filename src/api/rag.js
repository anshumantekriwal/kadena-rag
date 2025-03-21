import { Router } from 'express';

export default function createRagRouter(rag) {
  const router = Router();

  // Main RAG endpoint
  router.post('/rag', async (req, res) => {
    try {
      const { question, numResults = 5 } = req.body;
      
      if (!question) {
        return res.status(400).json({ error: 'Question is required' });
      }

      const response = await rag.query(question, numResults);
      
      res.json({
        question,
        answer: response.answer,
        sources: response.sources
      });
    } catch (error) {
      console.error('RAG query error:', error);
      res.status(500).json({ 
        error: 'Failed to process question',
        details: error.message
      });
    }
  });

  // Health check endpoint
  router.get('/rag/health', (req, res) => {
    res.json({ 
      status: 'ok',
      version: process.env.npm_package_version,
      services: {
        vectorDB: 'connected',
        openAI: 'connected'
      }
    });
  });

  return router;
}
