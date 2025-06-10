from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from storage_and_retrieval import embedding_model
from langchain.chains import LLMChain

# model name to use for text generation
model_name = "google/flan-t5-base"

# Load the tokenizer and model from HuggingFace Transformers
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Creating a text2text-generation pipeline with the loaded model and tokenizer
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512 
)

llm = HuggingFacePipeline(pipeline=pipe)

def trim_context_to_token_limit(docs, tokenizer, max_tokens:int):
    """
    Trims the provided documents so that their combined token count does not exceed max_tokens.

    Args:
        docs (list): List of documents, each with or without a 'page_content' attribute or as a string.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to count tokens.
        max_tokens (int): Maximum allowed number of tokens in the combined context.

    Returns:
        str: Concatenated document texts, trimmed to fit within the token limit.
    """
    context = ""
    total_tokens = 0

    for doc in docs:
        doc_text = doc.page_content.strip() if hasattr(doc, 'page_content') else str(doc)
        doc_tokens = tokenizer(doc_text, return_tensors='pt', truncation=False)['input_ids'][0]
        if total_tokens + len(doc_tokens) <= max_tokens:
            context += doc_text + "\n\n"
            total_tokens += len(doc_tokens)
        else:
            break
    return context.strip()

def get_query_response(query:str)-> str:
    """
    Processes a user query by retrieving relevant documents, constructing a prompt,
    and generating a response using a language model.

    Args:
        query (str): The user's question.

    Returns:
        str: The generated answer based on retrieved context.
    """
    from storage_and_retrieval import db
    from storage_and_retrieval import retriever
    results = retriever.get_relevant_documents(query)
    context = trim_context_to_token_limit(results, tokenizer, 512)
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Use the following context to answer the question at the end. 
        Even if the question is not directly answered in the context, say "It's not clearly mentioned but my best guess is"
        and use the context to provide a guess.
        Give a detailed answer based on the context provided.
        Context:
        {context}

        Question:
        {question}

        Answer:"""
    )
    print("Context length:", len(context.split()))

    rag_chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )
    print("Running RAG chain...")
    response = rag_chain.run({
        "context": context,
        "question": query
    })

    return response


