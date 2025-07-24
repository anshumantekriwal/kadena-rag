#!/usr/bin/env python3
"""
Comprehensive Kadena Documentation Data Extractor

This script implements all the principles and improvements for extracting 
documentation data from the Kadena docs repository:

PRINCIPLES IMPLEMENTED:
1. Remove YAML frontmatter (no repetitive metadata)
2. Start content from description/main content (not title)
3. Remove all \n characters for continuous text flow
4. Clean and optimize content format for RAG
5. Maintain comprehensive coverage (295+ documents)
6. Keep existing JSON structure (content, source, title)
7. Extract clean titles from frontmatter or headings
8. Optimize for embedding and retrieval

IMPROVEMENTS:
- Better content cleaning (remove markdown artifacts)
- Normalize spacing and punctuation
- Remove redundant formatting
- Clean up code blocks and tables for better readability
- Extract meaningful descriptions
- Category-based processing for different doc types
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List

class KadenaDocsExtractor:
    def __init__(self, repo_path: str = "kadena-docs/docs", output_path: str = "data/docs.json"):
        self.repo_path = Path(repo_path)
        self.output_path = Path(output_path)
        
    def extract_metadata_from_frontmatter(self, content: str) -> Dict[str, str]:
        """Extract metadata from YAML frontmatter."""
        metadata = {}
        
        # Extract frontmatter
        frontmatter_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            
            # Extract title
            title_match = re.search(r'^title:\s*(.+)$', frontmatter, re.MULTILINE)
            if title_match:
                metadata['title'] = re.sub(r'^["\']|["\']$', '', title_match.group(1).strip())
            
            # Extract description
            desc_match = re.search(r'^description:\s*(.+?)(?=^\w+:|$)', frontmatter, re.MULTILINE | re.DOTALL)
            if desc_match:
                desc = desc_match.group(1).strip()
                desc = re.sub(r'\n\s*', ' ', desc)  # Join multi-line descriptions
                desc = re.sub(r'^["\']|["\']$', '', desc)  # Remove quotes
                metadata['description'] = desc
                
        return metadata
    
    def extract_title_fallback(self, content: str) -> str:
        """Extract title from content if not in frontmatter."""
        # Remove frontmatter first
        content_no_frontmatter = re.sub(r'^---\n.*?\n---\n\n', '', content, flags=re.DOTALL)
        
        # Look for first heading
        heading_match = re.search(r'^#\s+(.+)$', content_no_frontmatter, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()
            
        return "Untitled"
    
    def clean_content(self, content: str) -> str:
        """Clean and optimize content for RAG processing."""
        
        # Remove YAML frontmatter
        content = re.sub(r'^---\n.*?\n---\n\n', '', content, flags=re.DOTALL)
        
        # Clean up markdown tables - convert to more readable format
        content = self.clean_markdown_tables(content)
        
        # Clean up code blocks - preserve but make more readable
        content = self.clean_code_blocks(content)
        
        # Remove excessive markdown formatting
        content = self.clean_markdown_formatting(content)
        
        # Normalize spacing and punctuation
        content = self.normalize_spacing(content)
        
        # Remove newlines for continuous flow
        content = content.replace('\n', ' ')
        
        # Clean up multiple spaces
        while '  ' in content:
            content = content.replace('  ', ' ')
            
        return content.strip()
    
    def clean_markdown_tables(self, content: str) -> str:
        """Convert markdown tables to more readable format."""
        # Find table patterns and convert them
        def replace_table(match):
            table_content = match.group(0)
            lines = table_content.split('\n')
            
            # Skip separator lines (those with just | --- | --- |)
            content_lines = [line for line in lines if not re.match(r'^\s*\|[\s\-\|]+\|\s*$', line)]
            
            # Convert to more readable format
            readable_lines = []
            for line in content_lines:
                if '|' in line:
                    # Clean up table cells
                    cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                    if cells:
                        readable_lines.append(' | '.join(cells))
            
            return ' '.join(readable_lines)
        
        # Match markdown tables
        table_pattern = r'(\|[^\n]+\|\n)+(\|[\s\-\|]+\|\n)?(\|[^\n]+\|\n)+'
        content = re.sub(table_pattern, replace_table, content, flags=re.MULTILINE)
        
        return content
    
    def clean_code_blocks(self, content: str) -> str:
        """Clean up code blocks while preserving essential information."""
        # Replace code blocks with cleaner format
        def replace_code_block(match):
            lang = match.group(1) or ""
            code = match.group(2).strip()
            
            # For very long code blocks, truncate and add summary
            if len(code) > 500:
                code = code[:500] + "... [code continues]"
            
            return f"Code example ({lang}): {code}"
        
        # Match fenced code blocks
        content = re.sub(r'```(\w+)?\n(.*?)\n```', replace_code_block, content, flags=re.DOTALL)
        
        # Clean inline code
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        return content
    
    def clean_markdown_formatting(self, content: str) -> str:
        """Remove excessive markdown formatting."""
        # Remove markdown links but keep text
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        
        # Remove bold/italic formatting
        content = re.sub(r'\*\*([^\*]+)\*\*', r'\1', content)
        content = re.sub(r'\*([^\*]+)\*', r'\1', content)
        content = re.sub(r'__([^_]+)__', r'\1', content)
        content = re.sub(r'_([^_]+)_', r'\1', content)
        
        # Clean up headers - convert to simple text
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        
        # Remove HTML entities
        content = re.sub(r'&nbsp;', ' ', content)
        content = re.sub(r'&[a-zA-Z]+;', '', content)
        
        return content
    
    def normalize_spacing(self, content: str) -> str:
        """Normalize spacing and punctuation."""
        # Fix spacing around punctuation
        content = re.sub(r'\s+([,.;:!?])', r'\1', content)
        content = re.sub(r'([,.;:!?])\s*', r'\1 ', content)
        
        # Normalize quotes - fixed regex patterns
        content = re.sub(r'["""]', '"', content)
        content = re.sub(r"[''']", "'", content)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)
        
        return content
    
    def categorize_document(self, source_path: str) -> str:
        """Categorize document based on its path."""
        path_parts = source_path.split('/')
        
        if 'api' in path_parts:
            return 'api'
        elif 'pact-5' in path_parts:
            return 'pact-functions'
        elif 'smart-contracts' in path_parts:
            return 'smart-contracts'
        elif 'guides' in path_parts:
            return 'guides'
        elif 'reference' in path_parts:
            return 'reference'
        else:
            return 'general'
    
    def process_markdown_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            # Extract metadata
            metadata = self.extract_metadata_from_frontmatter(raw_content)
            
            # Get title (from metadata or fallback)
            title = metadata.get('title') or self.extract_title_fallback(raw_content)
            
            # Clean content
            clean_content = self.clean_content(raw_content)
            
            # If we have a description, start with that
            if 'description' in metadata:
                # Start content with description, then main content
                description = metadata['description']
                # Remove the title from the beginning if it's redundant
                if clean_content.startswith(title):
                    clean_content = clean_content[len(title):].strip()
                clean_content = f"{description} {clean_content}"
            
            # Get relative path
            relative_path = file_path.relative_to(self.repo_path)
            source = str(relative_path).replace('\\', '/')
            
            # Categorize
            category = self.categorize_document(source)
            
            return {
                "content": clean_content,
                "source": source,
                "title": title,
                "category": category
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def collect_markdown_files(self) -> List[Path]:
        """Collect all markdown files from the docs directory."""
        markdown_files = []
        
        for root, dirs, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith('.md'):
                    file_path = Path(root) / file
                    markdown_files.append(file_path)
        
        return sorted(markdown_files)
    
    def extract_all_docs(self) -> List[Dict[str, Any]]:
        """Extract all documentation files."""
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path {self.repo_path} does not exist")
        
        print("Collecting markdown files...")
        markdown_files = self.collect_markdown_files()
        print(f"Found {len(markdown_files)} markdown files")
        
        docs_data = []
        
        print("Processing markdown files...")
        for i, file_path in enumerate(markdown_files):
            doc_entry = self.process_markdown_file(file_path)
            if doc_entry:
                docs_data.append(doc_entry)
                
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(markdown_files)} files...")
        
        print(f"Successfully processed {len(docs_data)} documents")
        return docs_data
    
    def save_docs(self, docs_data: List[Dict[str, Any]]) -> None:
        """Save the processed documentation data."""
        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(exist_ok=True)
        
        # Remove category field for final output (keep same structure as before)
        clean_docs = []
        for doc in docs_data:
            clean_doc = {
                "content": doc["content"],
                "source": doc["source"], 
                "title": doc["title"]
            }
            clean_docs.append(clean_doc)
        
        print(f"Writing to {self.output_path}...")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_docs, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved {len(clean_docs)} documents to {self.output_path}")
        
        # Print summary by category
        categories = {}
        for doc in docs_data:
            cat = doc.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nDocument categories:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} documents")

def main():
    """Main function to extract documentation data."""
    extractor = KadenaDocsExtractor()
    
    try:
        # Extract all documentation
        docs_data = extractor.extract_all_docs()
        
        # Save to file
        extractor.save_docs(docs_data)
        
        print("\n✅ Documentation extraction completed successfully!")
        print(f"📄 Total documents: {len(docs_data)}")
        print(f"💾 Output file: {extractor.output_path}")
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 