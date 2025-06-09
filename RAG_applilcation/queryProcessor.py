
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from storage_and_retrieval import embedding_model
from storage_and_retrieval import retriever
from storage_and_retrieval import db
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

def truncate_context(text, tokenizer, max_tokens=512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def get_query_response(query:str)-> str:
    results = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in results])
    context = truncate_context(context, tokenizer, max_tokens=512)
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


