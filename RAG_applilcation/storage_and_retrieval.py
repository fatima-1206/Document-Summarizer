
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
semantic_chunker = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")

db = Chroma( persist_directory="chroma_store", embedding_function=embedding_model)
retriever = db.as_retriever()

def reload_connection():
    global db, retriever
    db.delete_collection()
    db = Chroma( persist_directory="chroma_store", embedding_function=embedding_model)
    retriever = db.as_retriever()

def chunk_text(text:str, file_path:str):
    chunks = semantic_chunker.split_text(text)
    print("Chunked text into", len(chunks), "chunks.")
    print("Generating metadata for each chunk...")
    metadatas = [
        {
            "source": file_path,
            "chunk_index": i,
            "length": len(chunks[i])

        }
        for i in range(len(chunks))
    ]
    print("Metadata generated for all chunks.")
    return chunks, metadatas

def store_chunks(chunks, metadatas):
    print("Storing chunks in the database...")
    if not chunks:
        print("No chunks to store.")
        return
    if not metadatas:
        print("No metadata to store.")
        return
    db.add_texts( texts=chunks, metadatas=metadatas)
    db.persist()
    print(f"Stored {len(chunks)} chunks in the database...")

