# NoteBot — AI PDF Assistant

NoteBot is an AI-powered PDF Question Answering system that allows users to upload PDF notes and ask questions about their content.

Instead of manually searching through lengthy documents, NoteBot uses **Retrieval-Augmented Generation (RAG)** to find the most relevant information from the uploaded PDF and generate an answer using a **local AI model**.

## Features

*  Upload PDF notes or documents
*  Extract text from PDF files
*  Split documents into smaller text chunks
*  Generate semantic embeddings using Hugging Face
*  Perform similarity search using FAISS
*  Generate answers using the local `google/flan-t5-base` model
*  No OpenAI API key required
*  Simple Streamlit web interface
*  Runs locally on your machine

##  How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

```text
                PDF Document
                     │
                     ▼
              Text Extraction
                  PyPDF2
                     │
                     ▼
               Text Chunking
       RecursiveCharacterTextSplitter
                     │
                     ▼
             Text Embeddings
       all-MiniLM-L6-v2
                     │
                     ▼
               FAISS Vector Store
                     │
                     │
User Question ───────┤
                     ▼
             Similarity Retrieval
                     │
                     ▼
             Relevant Context
                     │
                     ▼
              FLAN-T5 Base
              Local LLM
                     │
                     ▼
                  Answer
```

##  Methodology

### 1. PDF Text Extraction

When a user uploads a PDF, NoteBot uses **PyPDF2** to extract the text from each page.

### 2. Text Chunking

Large documents are divided into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`.

The current configuration uses:

```text
Chunk Size: 1000 characters
Chunk Overlap: 200 characters
```

Chunking allows the system to work with smaller and more relevant portions of the document.

### 3. Embeddings

Each text chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These embeddings represent the semantic meaning of the text.

### 4. Vector Search

The generated embeddings are stored in a **FAISS** vector store.

When a user asks a question, the question is converted into an embedding and FAISS searches for the most semantically relevant document chunks.

### 5. Retrieval-Augmented Generation

The retrieved chunks are passed to the language model as context.

NoteBot uses:

```text
google/flan-t5-base
```

The model generates an answer based on the retrieved context rather than relying only on its pretrained knowledge.

### 6. Context Restriction

The prompt instructs the model to answer only from the provided context.

If the required information cannot be found, the system is instructed to respond:

```text
I don't know
```

This helps reduce unsupported answers.

##  Tech Stack

| Technology                     | Purpose                  |
| ------------------------------ | ------------------------ |
| Python                         | Application development  |
| Streamlit                      | Web interface            |
| PyPDF2                         | PDF text extraction      |
| LangChain                      | RAG pipeline             |
| RecursiveCharacterTextSplitter | Text chunking            |
| Hugging Face                   | Embeddings and local LLM |
| all-MiniLM-L6-v2               | Text embeddings          |
| FLAN-T5 Base                   | Local language model     |
| FAISS                          | Vector similarity search |
| Sentence Transformers          | Semantic embeddings      |

##  Project Structure

```text
NoteBot/
│
├── chatbox.py
├── requirements.txt
├── README.md
└── .gitignore

```


##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/kanha271/NoteBot.git
cd NoteBot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

##  Run the Application

Start the Streamlit application:

```bash
streamlit run chatbox.py
```

The application will open in your browser.

Upload a PDF and enter a question related to its content.

##  Example

Suppose the uploaded PDF contains notes about Machine Learning.

You can ask:

```text
What is supervised learning?
```

NoteBot retrieves the relevant section from the PDF and provides an answer using the retrieved context.

##  Local AI

One of the main goals of NoteBot is to avoid depending on paid external AI APIs.

The project uses:

```text
google/flan-t5-base
```

as a locally loaded language model and:

```text
sentence-transformers/all-MiniLM-L6-v2
```

for generating embeddings.

Therefore, no OpenAI API key is required.

##  Problem Solved

Reading and searching through large PDF documents manually can be time-consuming.

NoteBot provides a conversational interface where users can directly ask questions about their documents and receive answers based on the document's content.

##  Future Improvements

Possible improvements include:

* Support for multiple PDFs
* Persistent FAISS indexes
* Displaying the source page for each answer
* Conversation history
* Better document preprocessing
* Support for larger and more capable local LLMs
* Improved UI and document management
* OCR support for scanned PDFs
* Streaming model responses

##  Project

**NoteBot — AI PDF Assistant**

Built using Python, Streamlit, LangChain, Hugging Face and FAISS.
