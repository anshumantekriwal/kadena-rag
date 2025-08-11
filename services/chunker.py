import hashlib
from typing import Dict, Any, Iterable, List, Tuple

def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def window_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def flatten_kv(d: Dict[str, Any]) -> str:
    parts = []
    for k, v in d.items():
        if v is None or v == "":
            continue
        if isinstance(v, (dict, list)):
            parts.append(f"{k}: {flatten_kv(v) if isinstance(v, dict) else '; '.join(map(str, v))}")
        else:
            parts.append(f"{k}: {v}")
    return " | ".join(parts)

def stringify_links(links: Dict[str, str]) -> List[str]:
    out = []
    for k, v in (links or {}).items():
        if isinstance(v, str) and v.startswith("http"):
            out.append(v)
    return out

def flatten_ecosystem_content(title: str, content: Dict[str, Any]) -> List[Tuple[str, str, List[str]]]:
    """
    Returns list of tuples: (section, text, links)
    """
    items: List[Tuple[str, str, List[str]]] = []

    # overview
    ov = content.get("overview")
    if isinstance(ov, str) and ov.strip():
        items.append(("overview", ov.strip(), stringify_links(content.get("links", {}))))

    # key_features
    kf = content.get("key_features")
    if isinstance(kf, list) and kf:
        items.append(("key_features", " • ".join([str(x) for x in kf]), stringify_links(content.get("links", {}))))

    # metrics / tokenomics / team / value_proposition
    for sec in ("metrics", "tokenomics"):
        val = content.get(sec)
        if isinstance(val, dict) and val:
            items.append((sec, flatten_kv(val), stringify_links(content.get("links", {}))))
    if isinstance(content.get("team"), list) and content["team"]:
        team_lines = []
        for t in content["team"]:
            if isinstance(t, dict):
                nm = t.get("name", "")
                rl = t.get("role", "")
                team_lines.append(f"{nm} — {rl}".strip(" —"))
            else:
                team_lines.append(str(t))
        items.append(("team", " | ".join(team_lines), stringify_links(content.get("links", {}))))

    vp = content.get("value_proposition")
    if isinstance(vp, str) and vp.strip():
        items.append(("value_proposition", vp.strip(), stringify_links(content.get("links", {}))))

    # links (store as text too for recall)
    ln = content.get("links") or {}
    if isinstance(ln, dict) and ln:
        items.append(("links", flatten_kv(ln), stringify_links(ln)))

    # catch-all for any other fields
    for k, v in content.items():
        if k in {"overview","key_features","metrics","tokenomics","team","value_proposition","links"}:
            continue
        if isinstance(v, (dict, list)) and v:
            items.append((k, flatten_kv(v) if isinstance(v, dict) else " • ".join(map(str, v)), stringify_links(content.get("links", {}))))
        elif isinstance(v, str) and v.strip():
            items.append((k, v.strip(), stringify_links(content.get("links", {}))))

    return items