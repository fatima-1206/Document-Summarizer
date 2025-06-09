
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
semantic_chunker = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")
db = Chroma( persist_directory="chroma_store", embedding_function=embedding_model)
retriever = db.as_retriever()

def chunk_text(text:str, file_path:str):
    chunks = semantic_chunker.split_text(text)
    metadatas = [
        {
            "source": file_path,
            "chunk_index": i,
            "length": len(chunks[i])

        }
        for i in range(len(chunks))
    ]
    return chunks, metadatas

def store_chunks(chunks, metadatas):
    chunks = list(set(chunks))  
    db.add_texts( texts=chunks, metadatas=metadatas)
    db.persist()
    print(f"Stored {len(chunks)} chunks in the database...")