"""import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Step 1: Setup LLM (Mistral with HiggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(HUGGINGFACE_REPO_ID, hf_token):
    llm=HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":512} 
    )
    return llm

# Step 2: Connect LLM with FAISS and Create Chain

DB_FAISS_PATH="vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE=
Use the pieces of information provided in the context to answer user's question.
If you dnt know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,
                           input_variables=["context", "question"])
    return prompt

# Load database
DB_FAISS_PATH="vectorstore/db_faiss"

embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm= load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k":3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )


#Now invokwe with single quary
user_query=input("Write Query Here: ")
response=qa_chain.invoke({"query":user_query})
print("RESULT: ", response['result'])
print("SOURCE DOCUMENTS: ", response['source_documents'])"""


import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA  # Updated import path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

# Step 1: Setup LLM (Mistral with HuggingFace)


from langchain_groq import ChatGroq

def load_llm():
    return ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0.5,
        max_tokens=1024,
        reasoning_format="hidden",  # keeps the thinking out of the answer
        api_key=os.environ.get("GROQ_API_KEY"),
    )

# Step 2: Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# Load database
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain (Passed both repo_id and HF_TOKEN here)
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with single query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})
print("RESULT: ", response['result'])
print("SOURCE DOCUMENTS: ", response['source_documents'])