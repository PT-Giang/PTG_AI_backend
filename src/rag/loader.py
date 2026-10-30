from pathlib import Path

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document


def load_document(file_path: str | Path, source_name: str | None = None) -> list[Document]:
    path = Path(file_path)
    if path.suffix.lower() not in {".pdf", ".docx"}:
        raise ValueError("Only .pdf and .docx documents are supported")
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")
    documents = [
        document for document in UnstructuredFileLoader(
            str(path), strategy="fast", languages=["vie", "eng"]
        ).load()
        if document.page_content.strip()
    ]
    if source_name:
        display_name = Path(source_name.replace("\\", "/")).name
        for document in documents:
            document.metadata["source_name"] = display_name
    if not documents:
        raise ValueError("Document contains no extractable text")
    return documents
