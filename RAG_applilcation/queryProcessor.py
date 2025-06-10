
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from storage_and_retrieval import embedding_model
from langchain.chains import LLMChain

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512
)

llm = HuggingFacePipeline(pipeline=pipe)

def trim_context_to_token_limit(docs, tokenizer, max_tokens):
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


