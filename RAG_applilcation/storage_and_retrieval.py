
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Initialize the embedding model using HuggingFace's sentence-transformers
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create a semantic chunker to split text into semantically meaningful chunks
# The breakpoint_threshold_type="percentile" controls how chunk boundaries are determined
semantic_chunker = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")

# Initialize the Chroma vector database with the embedding model for storing and retrieving embeddings
db = Chroma(persist_directory="chroma_store", embedding_function=embedding_model)

# Create a retriever object from the database for similarity search
retriever = db.as_retriever()



def reload_connection():
    """Reloads the connection to the Chroma vector database and resets the retriever."""
    global db, retriever
    db.delete_collection()
    db = Chroma( persist_directory="chroma_store", embedding_function=embedding_model)
    retriever = db.as_retriever()



def chunk_text(text:str, file_path:str):
    """
    Splits the input text into semantically meaningful chunks and generates metadata for each chunk.

    Args:
        text (str): The text to be chunked.
        file_path (str): The path to the source file from which the text was extracted.

    Returns:
        tuple: A tuple containing a list of text chunks and a list of corresponding metadata dictionaries.
    """
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
    """
    Stores the provided text chunks and their metadata in the Chroma vector database.

    Args:
        chunks (list of str): The list of text chunks to store.
        metadatas (list of dict): The list of metadata dictionaries corresponding to each chunk.
    """
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

