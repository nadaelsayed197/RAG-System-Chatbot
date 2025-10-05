import os
from dotenv import load_dotenv
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import Groq
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = Groq(model="llama3-8b-8192", groq_api_key=groq_api_key)

# Load data
with open("data/courses.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_text(text)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vectorstore
vectorstore = Chroma.from_texts(texts, embedding=embeddings)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create RAG pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
)

# Streamlit UI
st.title("Course Info Chatbot (RAG System)")
st.markdown("Ask me anything about the available courses!")

user_query = st.text_input("Your question:")

if user_query:
    with st.spinner("Thinking..."):
        result = qa.run(user_query)
        st.success(result)
