import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline


# ============================================================
# NoteBot - AI PDF Assistant
# ============================================================

st.set_page_config(
    page_title="NoteBot - AI PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 NoteBot - AI PDF Assistant")
st.caption("Ask questions about your PDF using a local RAG pipeline.")


# ============================================================
# Load Local AI Models
# ============================================================

@st.cache_resource(show_spinner="Loading FLAN-T5 model...")
def load_llm():
    model_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_new_tokens=256
    )

    return HuggingFacePipeline(
        pipeline=model_pipeline
    )


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


llm = load_llm()
embeddings = load_embeddings()


# ============================================================
# PDF Functions
# ============================================================

def extract_text_from_pdf(uploaded_file):
    """Extract readable text from all pages of a PDF."""
    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)

    return "\n".join(pages)


def split_text(text):
    """Split document text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)


def create_vectorstore(chunks):
    """Create a FAISS vector store from document chunks."""
    return FAISS.from_texts(
        chunks,
        embedding=embeddings
    )


# ============================================================
# RAG Prompt
# ============================================================

PROMPT_TEMPLATE = """
You are NoteBot, an AI assistant that answers questions about
uploaded PDF documents.

Use ONLY the information provided in the context.

If the answer cannot be found in the context, say:
"I don't know"

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""


def create_qa_chain(vectorstore):
    """Create the retrieval and generation chain."""

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("📂 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# ============================================================
# Home Screen
# ============================================================

if uploaded_file is None:

    st.info("👈 Upload a PDF from the sidebar to get started.")

    st.markdown("""
    ### How NoteBot works

    1. **Upload** a PDF.
    2. **Extract** text using PyPDF2.
    3. **Split** text into smaller chunks.
    4. **Create embeddings** using `all-MiniLM-L6-v2`.
    5. **Store embeddings** in FAISS.
    6. **Retrieve relevant chunks** for your question.
    7. **Generate an answer** using local `FLAN-T5`.
    """)


# ============================================================
# Process Uploaded PDF
# ============================================================

else:

    file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    # Process only when a new PDF is uploaded
    if st.session_state.get("processed_file") != file_key:

        with st.spinner("Reading PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)

        if not extracted_text.strip():
            st.error(
                "No readable text was found in this PDF. "
                "The PDF may be scanned or image-based."
            )
            st.stop()

        with st.spinner("Splitting document into chunks..."):
            chunks = split_text(extracted_text)

        with st.spinner("Creating FAISS vector index..."):
            vectorstore = create_vectorstore(chunks)

        st.session_state.processed_file = file_key
        st.session_state.vectorstore = vectorstore
        st.session_state.chunk_count = len(chunks)

    else:
        vectorstore = st.session_state.vectorstore


    # ========================================================
    # PDF Information
    # ========================================================

    st.sidebar.success("PDF ready!")

    st.sidebar.write(
        f"**File:** {uploaded_file.name}"
    )

    st.sidebar.write(
        f"**Chunks:** {st.session_state.chunk_count}"
    )


    # ========================================================
    # Question Answering
    # ========================================================

    st.subheader("💬 Ask a question")

    question = st.text_input(
        "What would you like to know about this PDF?",
        placeholder="Example: What is supervised learning?"
    )


    if question.strip():

        with st.spinner(
            "Searching the document and generating an answer..."
        ):

            try:
                qa_chain = create_qa_chain(vectorstore)

                response = qa_chain.invoke(
                    {"query": question}
                )

                answer = response["result"]

            except Exception as error:
                st.error(f"An error occurred: {error}")
                st.stop()


        st.subheader("📝 Answer")
        st.write(answer)


        # ====================================================
        # Show Retrieved Context
        # ====================================================

        source_documents = response.get(
            "source_documents",
            []
        )

        if source_documents:

            with st.expander(
                f"🔎 Retrieved Context ({len(source_documents)} chunks)"
            ):

                for index, document in enumerate(
                    source_documents,
                    start=1
                ):

                    st.markdown(
                        f"**Retrieved Chunk {index}**"
                    )

                    st.write(
                        document.page_content
                    )

                    if index < len(source_documents):
                        st.divider()
