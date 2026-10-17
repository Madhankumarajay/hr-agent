import re
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
BASE = "https://raw.githubusercontent.com/Azure-Samples/azure-search-openai-demo/main/data/"
PDFS = ["employee_handbook.pdf", "Northwind_Standard_Benefits_Details.pdf"]

_col = None


def _download(name: str) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    if not path.exists():
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "hr-agent"})
        path.write_bytes(urllib.request.urlopen(req, timeout=30).read())
    return path


def init() -> int:
    global _col
    import chromadb
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader

    col = chromadb.PersistentClient(path=str(DATA / "chroma")).get_or_create_collection("handbook")
    if col.count() == 0:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        ids, docs, metas = [], [], []
        for name in PDFS:
            try:
                reader = PdfReader(str(_download(name)))
            except Exception as e:
                print(f"[rag] skipping {name}: {e}")
                continue
            for page_no, page in enumerate(reader.pages, start=1):
                text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
                for j, chunk in enumerate(splitter.split_text(text)):
                    ids.append(f"{name}:{page_no}:{j}")
                    docs.append(chunk)
                    metas.append({"source": name, "page": page_no})
        if not docs:
            raise RuntimeError("No PDF text indexed. Put the PDFs in backend/data/ and restart.")
        for i in range(0, len(docs), 100):
            col.add(ids=ids[i:i + 100], documents=docs[i:i + 100], metadatas=metas[i:i + 100])
    _col = col
    return col.count()


def search(query: str, k: int = 4) -> list:
    res = _col.query(query_texts=[query], n_results=k)
    return [
        {"text": d, "source": m["source"], "page": m["page"]}
        for d, m in zip(res["documents"][0], res["metadatas"][0])
    ]
