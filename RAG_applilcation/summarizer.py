from storage_and_retrieval import db as vectorstore
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
import numpy as np

summarization_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
summarization_model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer_sum = AutoTokenizer.from_pretrained(summarization_model_name)
model_sum = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)
pipe_sum = pipeline(
    "text2text-generation",
    model=model_sum,
    tokenizer=tokenizer_sum,
    max_length=1024
)

# Wrap it with LangChain
llm_sum = HuggingFacePipeline(pipeline=pipe_sum)

def get_summary(vectorstore = vectorstore, llm_sum = llm_sum, k=5):

    collection = vectorstore._collection
    all_docs = collection.get(include=["documents", "embeddings"])
    embeddings = np.array(all_docs["embeddings"])
    documents = all_docs["documents"]

    centroid = np.mean(embeddings, axis=0).reshape(1, -1)
    salience_scores = np.linalg.norm(embeddings - centroid, axis=1)

    # Compute salience bounds
    lower_bound = np.percentile(salience_scores, 33)   # Avoid central
    upper_bound = np.percentile(salience_scores, 85)   # Avoid outliers

    # Filter chunks in the mid-salience band
    mid_band_indices = [i for i, score in enumerate(salience_scores)
                        if lower_bound < score < upper_bound]

    # Sort those chunks by descending salience
    mid_band_indices.sort(key=lambda i: salience_scores[i], reverse=False)

    # De-duplicate while preserving order
    seen = set()
    unique_indices = []
    for idx in mid_band_indices:
        doc = documents[idx]
        if doc not in seen:
            seen.add(doc)
            unique_indices.append(idx)
        if len(unique_indices) == k:
            break

    salient_chunks = [documents[i] for i in unique_indices]

    # Quality check on chunks
    def is_valid_chunk(chunk):
        return len(chunk.split()) > 10 and "." in chunk

    filtered_chunks = list(filter(is_valid_chunk, salient_chunks))
    text = "\n\n".join(filtered_chunks)

    if len(text) > 3000:
        text = text[:3000]
    print(f"Generating summary for {len(filtered_chunks)} chunks...")
    summary = llm_sum.invoke(text)
    print("Summary generated successfully.")
    return summary
