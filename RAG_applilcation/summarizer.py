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

def get_summary(k=5):
    
    collection = vectorstore._collection
    all_docs = collection.get(include=["documents", "embeddings"])

    if all_docs["embeddings"] is None or len(all_docs["embeddings"]) == 0:
        raise ValueError("No embeddings found in the vectorstore!")

    embeddings = np.array(all_docs["embeddings"])
    documents = all_docs["documents"]

    # Computing the centroid of all embeddings
    centroid = np.mean(embeddings, axis=0).reshape(1, -1)

    salience_scores = np.linalg.norm(embeddings - centroid, axis=1)

    # Get indices of the top-k most salient chunks
    top_k_indices = salience_scores.argsort()[::-1][:k]
    salient_chunks = [documents[i] for i in top_k_indices]

    input_text = "You are an academic writing assistant. Summarize the following document in elegant, natural language. Make sure it reads smoothly and sounds professional. Avoid copying bullet points verbatim. Use proper grammar and punctuation.\n"
    input_text = " ".join(salient_chunks)
    if len(input_text) > 3000:  
        input_text = input_text[:3000]

    summary = llm_sum.invoke(input_text)
    return summary