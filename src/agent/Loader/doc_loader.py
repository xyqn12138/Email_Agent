import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
from agent.utils.path_handler import get_absolute_path

class DocumentLoader:
    def load(self, file_path: str):
        file_path = get_absolute_path(file_path)
        filename = os.path.basename(file_path).lower()

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            doc_type = "PDF"
        elif filename.endswith((".docx", ".doc")):
            loader = Docx2txtLoader(file_path)
            doc_type = "Word"
        elif filename.endswith((".xlsx", ".xls")):
            loader = UnstructuredExcelLoader(file_path)
            doc_type = "Excel"
        else:
            raise ValueError(f"Unsupported file type: {filename}")

        raw_docs = loader.lazy_load()

        results = []
        for doc in raw_docs:
            results.append({
                "text": (doc.page_content or "").strip(),
                "filename": filename,
                "file_path": file_path,
                "file_type": doc_type,
                "page_number": doc.metadata.get("page", 0),
            })
        return results
    

if __name__ == "__main__":
    loader = DocumentLoader()
    docs = loader.load(r"src\agent\data\test.pdf")
    print(docs[0])