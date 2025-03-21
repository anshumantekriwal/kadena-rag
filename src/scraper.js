import * as cheerio from 'cheerio';
import fs from 'fs-extra';
import path from 'path';
import MarkdownIt from 'markdown-it';
import { glob } from 'glob';

const md = new MarkdownIt();
const OUTPUT_DIR = path.join(process.cwd(), 'data');

async function readDocContent(filePath) {
  try {
    const content = await fs.readFile(filePath, 'utf8');
    return content;
  } catch (error) {
    console.error(`Error reading ${filePath}:`, error.message);
    return null;
  }
}

async function processMarkdown(content, filePath) {
  // Convert markdown to text while preserving structure
  const html = md.render(content);
  const $ = cheerio.load(html);
  
  // Remove code blocks and HTML comments
  $('pre').remove();
  $('code').remove();
  
  // Get clean text
  const text = $.text()
    .replace(/\s+/g, ' ')
    .trim();

  return {
    content: text,
    source: filePath,
    title: path.basename(filePath, '.md')
  };
}

async function scrapeKadenaDocs() {
  const DOCS_DIR = path.join(process.cwd(), 'kadena-docs', 'docs');
  try {
    // Ensure output directory exists
    await fs.ensureDir(OUTPUT_DIR);
    
    console.log('Starting document processing...');
    const documents = [];
    
    // Find all markdown files in the docs directory
    const files = await glob('**/*.md', { cwd: DOCS_DIR });
    console.log(`Found ${files.length} markdown files`);

    for (const file of files) {
      const filePath = path.join(DOCS_DIR, file);
      console.log(`Processing: ${file}`);
      try {
        const content = await readDocContent(filePath);
        if (content) {
          const processed = await processMarkdown(content, file);
          console.log(`Successfully processed: ${file}`);
          documents.push(processed);
        }
      } catch (error) {
        console.error(`Error processing ${file}:`, error.message);
      }
    }

    if (documents.length === 0) {
      throw new Error('No documents were successfully processed');
    }

    // Save the processed documents
    await fs.writeJSON(path.join(OUTPUT_DIR, 'processed_docs.json'), documents, { spaces: 2 });
    console.log(`Processed ${documents.length} documents`);
    
    return documents;
  } catch (error) {
    console.error('Error scraping docs:', error);
    throw error;
  }
}

export default scrapeKadenaDocs;
