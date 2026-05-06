from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader

from agent.rag.Loader.base_loader import BaseLoader


class DocumentLoader(BaseLoader):
    document_type = "Document"
    supported_suffixes = (".pdf", ".docx", ".doc", ".xlsx", ".xls")

    def load(self, file_path: str):
        absolute_path = self.validate_path(file_path)
        filename = absolute_path.name.lower()

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(str(absolute_path))
            doc_type = "PDF"
        elif filename.endswith((".docx", ".doc")):
            loader = Docx2txtLoader(str(absolute_path))
            doc_type = "Word"
        elif filename.endswith((".xlsx", ".xls")):
            loader = UnstructuredExcelLoader(str(absolute_path))
            doc_type = "Excel"
        else:
            raise ValueError(f"Unsupported file type: {filename}")

        raw_docs = loader.lazy_load()
        results = []
        for doc in raw_docs:
            results.append(
                self.build_metadata(
                    absolute_path,
                    text=(doc.page_content or "").strip(),
                    file_type=doc_type,
                    page_number=doc.metadata.get("page", 0),
                )
            )
        return results


if __name__ == "__main__":
    loader = DocumentLoader()
    docs = loader.load(r"src\agent\data\test.pdf")
    print(docs[0])
