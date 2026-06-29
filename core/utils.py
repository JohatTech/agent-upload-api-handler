from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document as LCDocument
import re
from docx import Document 


class LightweightDocxLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[LCDocument]:
        import docx
        doc = docx.Document(self.file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        
        # Include table cells if present
        for table in doc.tables:
            for row in table.rows:
                row_vals = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_vals:
                    paragraphs.append(" | ".join(row_vals))
                    
        text = "\n".join(paragraphs)
        return [LCDocument(page_content=text, metadata={"source": self.file_path})]


class LightweightExcelLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[LCDocument]:
        import openpyxl
        wb = openpyxl.load_workbook(self.file_path, read_only=True, data_only=True)
        sheets_text = []
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            sheets_text.append(f"--- Sheet: {sheet_name} ---")
            for row in sheet.iter_rows(values_only=True):
                row_vals = [str(val).strip() if val is not None else "" for val in row]
                if any(row_vals):
                    sheets_text.append(" | ".join(row_vals))
        text = "\n".join(sheets_text)
        return [LCDocument(page_content=text, metadata={"source": self.file_path})]


def format_bytes(bytes_size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def load_doc_input(path, doc_type, chunk_size, chunk_overlap):
    if doc_type == "xlsx":
       loader = LightweightExcelLoader(path)
    elif doc_type == "docx":
       loader = LightweightDocxLoader(path)
    elif doc_type == "pdf":
         loader = PyMuPDFLoader(path, mode="page", pages_delimiter=" ")
    elif doc_type == "txt":
         loader = TextLoader(path)
    else:
         raise ValueError(f"Unsupported doc_type: {doc_type}")
        
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return documents, chunks 


def get_text_file(example_path):
    with open(example_path, encoding="utf8") as file:
        data = file.readlines()
    texts = " ".join(data)
    return texts

def write_report(text, file_name):
    try:
        doc = Document()
        doc.add_paragraph(text)
        print(f"writing file name: {file_name}")
        doc.save(file_name)
    except BaseException as e:
        print(f"error writing report: {e}")

def set_collection_name(name):
    name = name.lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = name.strip("_")
    return name
