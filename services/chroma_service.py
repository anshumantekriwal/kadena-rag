from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import json, os, logging, time
from services.chunker import window_chunks, flatten_ecosystem_content, hash_text, stringify_links

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_metadata(md):
    return {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in md.items()}

class ChromaService:
    def __init__(self):
        load_dotenv()
        self.persist_directory = "./chromadb"
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
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
        logger.info("Initializing/attaching Chroma collections")
        self._initialize_or_update()

    # ---------- indexing ----------

    def _initialize_or_update(self) -> None:
        os.makedirs(self.persist_directory, exist_ok=True)
        for collection_key, cfg in self.collections.items():
            name = cfg["name"]
            vs = Chroma(collection_name=name,
                        embedding_function=self.embedding_function,
                        persist_directory=self.persist_directory)
            # Upsert documents for each source file
            for src in cfg["sources"]:
                self._upsert_source(vs, collection_key, src)

    def _read_json(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} must be a JSON array")
        return data

    def _upsert_source(self, vector_store: Chroma, collection_key: str, src: Dict[str, str]) -> None:
        path, data_type = src["file"], src["type"]
        if not os.path.exists(path):
            logger.warning("Missing data file: %s", path); return

        records = self._read_json(path)
        docs, metadatas, ids = [], [], []

        for idx, rec in enumerate(records):
            title = rec.get("title", "")
            source = rec.get("source", "unknown")
            raw_content = rec.get("content")

            # ---- handle ecosystem nested JSON vs plain text ----
            if collection_key == "ecosystem" and isinstance(raw_content, dict):
                # sectioned flattening
                for section, text, links in flatten_ecosystem_content(title, raw_content):
                    for chunk_text in window_chunks(text):
                        doc_id = hash_text(f"{path}|{title}|{section}|{chunk_text}")
                        docs.append(Document(page_content=chunk_text))
                        metadatas.append({
                            "collection": collection_key,
                            "source_file": path,
                            "data_type": data_type,
                            "title": title,
                            "section": section,
                            "links": json.dumps(links),
                            "hash": doc_id
                        })
                        ids.append(doc_id)
            else:
                # plain text sources
                if isinstance(raw_content, dict):
                    # very defensive: stringify if someone left JSON here
                    text = json.dumps(raw_content, ensure_ascii=False)
                else:
                    text = str(raw_content)
                for chunk_text in window_chunks(text):
                    doc_id = hash_text(f"{path}|{title}|{chunk_text}")
                    docs.append(Document(page_content=chunk_text))
                    metadatas.append({
                        "collection": collection_key,
                        "source_file": path,
                        "data_type": data_type,
                        "title": title,
                        "section": "content",
                        "links": json.dumps([]),
                        "hash": doc_id
                    })
                    ids.append(doc_id)

        if not docs:
            return

        # Filter metadata to ensure all values are primitive types
        metadatas = [filter_metadata(md) for md in metadatas]

        # Idempotent upsert (Chroma add with same ids is safe to re-run in practice)
        logger.info("Upserting %d chunks into %s from %s", len(docs), vector_store._collection.name, path)
        vector_store.add_texts(
            texts=[doc.page_content for doc in docs],
            metadatas=metadatas,
            ids=ids
        )
    # ---------- search ----------

    def search(self, query: str, top_k: int = 12, per_collection: int = 4) -> Tuple[List[Dict[str, Any]], List[dict]]:
        """
        Search across collections and return normalized chunks with metadata.
        """
        results: List[Dict[str, Any]] = []

        for collection_key, cfg in self.collections.items():
            name = cfg["name"]
            vs = Chroma(collection_name=name,
                        embedding_function=self.embedding_function,
                        persist_directory=self.persist_directory)

            retriever = vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": per_collection, "fetch_k": min(50, per_collection * 5), "lambda_mult": 0.25}
            )
            try:
                docs = retriever.invoke(query)
            except Exception as e:
                logger.error("Retriever error for %s: %s", name, e); continue

            for d in docs:
                md = dict(d.metadata or {})
                results.append({
                    "id": md.get("hash") or hash_text(d.page_content)[:16],
                    "text": d.page_content,
                    "title": md.get("title", ""),
                    "collection": md.get("collection", ""),
                    "source_file": md.get("source_file", ""),
                    "section": md.get("section", ""),
                    "links": md.get("links", []),
                    "score": None,
                    "metadata": md
                })

        # diversity + cap
        seen_ids, out = set(), []
        for r in results:
            if r["id"] in seen_ids:
                continue
            seen_ids.add(r["id"])
            out.append(r)
            if len(out) >= top_k:
                break

        return out, [r["metadata"] for r in out]

    # ---------- stats ----------

    def get_collection_stats(self) -> Dict[str, Any]:
        stats = {"total_collections": len(self.collections), "collections": {}}
        for key, cfg in self.collections.items():
            name = cfg["name"]
            try:
                vs = Chroma(collection_name=name,
                            embedding_function=self.embedding_function,
                            persist_directory=self.persist_directory)
                sample = vs.similarity_search("kadena", k=50)
                stats["collections"][key] = {
                    "collection_name": name,
                    "sample_size": len(sample),
                    "avg_len": sum(len(d.page_content) for d in sample) / max(1, len(sample)),
                }
            except Exception as e:
                stats["collections"][key] = {"error": str(e)}
        return stats