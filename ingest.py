import os
import glob
from typing import List
from dotenv import load_dotenv
import argparse

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS



class IngestConfiguration:
    def __init__(self):
        load_dotenv()

        self.persist_directory = os.environ.get('PERSIST_DIRECTORY')
        self.source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
        self.embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
        self.chunk_size = 500
        self.chunk_overlap = 50

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    return [load_single_document(file_path) for file_path in all_files]


def main(collection):
    # Load environment variables
    config = IngestConfiguration()
    persist_directory = config.persist_directory
    source_directory = config.source_directory
    embeddings_model_name = config.embeddings_model_name
    os.makedirs(source_directory, exist_ok=True)

    # Load documents and split in chunks
    print(f"Loading documents from {source_directory}")
    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap
    documents = load_documents(source_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    
    # Create and store locally vectorstore
    db = Chroma.from_documents(texts, embeddings, collection_name=collection, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    # try:
    db.persist()
    print("Persist operation completed successfully.\n")

    db = None


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", help="Saves the embedding in a collection name as specified")

    # Parse the command-line arguments
    args = parser.parse_args()

    main(args.collection)
