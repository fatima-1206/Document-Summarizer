from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from queryProcessor import trim_context_to_token_limit
import numpy as np

# Load the pre-trained summarization model from HuggingFace
summarization_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

# Specify the model name for tokenizer and pipeline usage
summarization_model_name = "sshleifer/distilbart-cnn-12-6"

# Load the tokenizer corresponding to the summarization model
tokenizer_sum = AutoTokenizer.from_pretrained(summarization_model_name)

# Load the model again for use in the pipeline (can reuse summarization_model if desired)
model_sum = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)

# Create a HuggingFace pipeline for text-to-text generation (summarization)
pipe_sum = pipeline(
    "text2text-generation",
    model=model_sum,
    tokenizer=tokenizer_sum,
    max_length=1024  # Set the maximum length for generated summaries
)

# Wrap the pipeline with LangChain's HuggingFacePipeline for integration with LangChain workflows
llm_sum = HuggingFacePipeline(pipeline=pipe_sum)

def get_summary(vectorstore="", llm_sum=llm_sum, k:int=20):
    """
    Generate a summary of the most salient document chunks from a vectorstore.

    Args:
        All arguments are optional.
        vectorstore (str, optional): The vectorstore to use for retrieving documents. 
        llm_sum (HuggingFacePipeline, optional): The language model pipeline for summarization. 
        k (int, optional): The number of salient chunks to consider for summarization. Defaults to 20.

    Returns:
        str: The generated summary of the selected document chunks.
    """
    from storage_and_retrieval import db
    vectorstore = db
    collection = vectorstore._collection
    all_docs = collection.get(include=["documents", "embeddings"])
    
    embeddings = np.array(all_docs["embeddings"])
    documents = all_docs["documents"]

    centroid = np.mean(embeddings, axis=0).reshape(1, -1)
    salience_scores = np.linalg.norm(embeddings - centroid, axis=1)

    lower_bound = np.percentile(salience_scores, 5)
    upper_bound = np.percentile(salience_scores, 86)

    mid_band_indices = [
        i for i, score in enumerate(salience_scores)
        if lower_bound < score < upper_bound
    ]

    mid_band_indices.sort(key=lambda i: salience_scores[i], reverse=True)
    k = min(k, len(mid_band_indices))
    salient_chunks = [documents[i] for i in mid_band_indices[:k]]
    # if a token is larger than 500 characters, get rid of it
    salient_chunks = [chunk for chunk in salient_chunks if len(chunk) <= 500]

    text = trim_context_to_token_limit(
        salient_chunks, tokenizer_sum, 1024
    )
    print(f"Number of chunks: {len(salient_chunks)}\n")
    summary = llm_sum.invoke(text)
    print(summary)
    return summary


