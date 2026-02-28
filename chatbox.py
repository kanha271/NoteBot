import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# Page configuration
st.set_page_config(page_title="NoteBot AI PDF Assistant")
st.title("NoteBot AI PDF Assistant")

# Load LLM with caching to avoid reloading on every interaction
#earlier i have used openAI but now i am using google/flan-t5-base which is free to use and can be run locally without API keys.
@st.cache_resource
def load_llm():
    pipe = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_new_tokens=256
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

if "llm" not in st.session_state:
    with st.spinner("Loading AI Model..."):
        st.session_state.llm = load_llm()

llm = st.session_state.llm

# PDF uploading 
st.sidebar.header("📂 Upload a PDF")
uploaded_file = st.sidebar.file_uploader(
    "Upload your notes",
    type="pdf"
)

if uploaded_file:

    pdf_reader = PdfReader(uploaded_file)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Text splitting, to break down the text into smaller chunks for better processing by the LLM
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    # embeddings, to convert the text chunks into vector representations that can be efficiently searched and retrieved by the LLM, here we are using a pre-trained model from HuggingFace called "sentence-transformers/all-MiniLM-L6-v2" which is a lightweight and efficient model for generating sentence embeddings. These embeddings will allow the LLM to understand the semantic meaning of the text chunks and retrieve relevant information when answering questions.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    # we are using FAISS (Facebook AI Similarity Search) to create a vector store from the text chunks and their corresponding embeddings. This allows us to efficiently search and retrieve relevant chunks of text based on the user's questions, enabling the LLM to provide accurate and contextually relevant answers.

    # Prompt template
    prompt_template = """
You are an intelligent AI assistant.

Answer ONLY using the provided context.

If the answer is not found in the context,
reply exactly with:
"I don't know"

Context:
{context}

Question:
{question}

Detailed Answer:
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Qa chain, we are creating a RetrievalQA chain that combines the LLM with the vector store retriever. This chain will take the user's question, retrieve relevant chunks of text from the vector store based on the embeddings, and then use the LLM to generate a detailed answer based on the retrieved context. The prompt template ensures that the LLM only answers using the provided context and handles cases where the answer is not found appropriately.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Question input
    question = st.text_input("Ask a question about your PDF")
    if question:

        with st.spinner("Thinking..."):
            result = qa_chain.run(question)

        st.success("Answer")
        st.write(result)