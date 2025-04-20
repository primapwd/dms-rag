# DMS-RAG: Document Management System with Retrieval-Augmented Generation

This project is a complete Retrieval-Augmented Generation (RAG) pipeline designed to work with local documents (PDFs). It extracts text, cleans it using advanced LLMs, indexes it in a vector database, and allows users to ask questions about the documents through a flexible chatbot interface.

## Overview

The core purpose of this project is to build a "Chat with your documents" system. It follows a multi-step pipeline to process raw documents and make them queryable.

The pipeline consists of the following stages:
1.  **Data Ingestion**: Starts with PDF documents placed in a designated folder.
2.  **OCR Processing**: Extracts raw text from PDF files using Tesseract OCR.
3.  **Text Cleansing**: Uses a Large Language Model (LLM) to clean and reformat the raw OCR text, removing artifacts like headers, footers, and page numbers while preserving the core content.
4.  **Document Chunking**: Splits the clean text into smaller, overlapping chunks suitable for embedding.
5.  **Embedding Generation**: Converts each text chunk into a vector embedding using a sentence-transformer model.
6.  **Database Population**: Stores the embeddings and their corresponding text metadata in a ChromaDB vector database for efficient similarity search.
7.  **Chat Interaction**: A chatbot CLI takes a user's query, retrieves the most relevant text chunks from the database, and uses an LLM to generate a concise answer based on the retrieved context.

## Features

-   **End-to-End RAG Pipeline**: From PDF to answer, all steps are included.
-   **Multi-Provider LLM Support**: Easily switch between **Google Gemini**, **OpenRouter**, and a local **Ollama** instance for both cleansing and question-answering.
-   **Local First**: All data (documents, text, embeddings, database) is stored locally.
-   **Extensible**: Each step in the pipeline is a separate, configurable Python script.
-   **PDF and Web Support**: Ingests information from both PDF files and web URLs.

## Project Structure

```
/
├─── documents/         # Place your source PDF files here in subfolders (e.g., documents/pdfs/)
├─── src/               # All Python source code for the pipeline
│    ├── ocr_processor.py
│    ├── text_cleaner.py
│    ├── doc_chunker.py
│    ├── embedding_generator.py
│    ├── database.py
│    └── chatbot.py
├─── db/                # ChromaDB vector database is stored here
├─── ocr_results/       # Raw text extracted from PDFs
├─── cleansed_results/  # Cleaned text after LLM processing
├─── chunked_results/   # JSON files containing text chunks
├─── embeddings/        # Stored embeddings (.npy) and metadata (.json)
├─── .env.example       # Example environment file
└─── requirements.txt   # Python dependencies
```

## Requirements

### System Dependencies

You need to install the following tools on your system:

-   **Python 3.10+**
-   **Tesseract-OCR**: For extracting text from images/PDFs.
-   **Poppler**: A PDF rendering library required by `pdf2image`.

**On macOS (using Homebrew):**
```bash
brew install python@3.10 tesseract poppler
```

**On Debian/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install -y python3.10-venv tesseract-ocr libpoppler-cpp-dev
```

### Python Environment

It is highly recommended to use a virtual environment.

-   **`venv`** (standard library)
-   **`uv`** (optional, but recommended for faster package installation)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd dms-rag
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Using venv
    python3 -m venv .venv
    source .venv/bin/activate

    # Or using uv
    uv venv
    source .venv/bin/activate
    ```

3.  **Install the Python dependencies:**
    ```bash
    # Using pip
    pip install -r requirements.txt

    # Or using uv
    uv pip install -r requirements.txt
    ```

## Configuration

1.  **Create a `.env` file** by copying the example file:
    ```bash
    cp .env.example .env
    ```

2.  **Edit the `.env` file** and add your API keys.
    -   `GOOGLE_API_KEY`: Required if you use the `google` provider.
    -   `OPENROUTER_API_KEY`: Required if you use the `openrouter` provider.
    -   You can also customize the default models and directory paths here.

3.  **For Ollama Users**: If you intend to use the `ollama` provider, ensure the Ollama service is running locally. You also need to pull the model you want to use, for example:
    ```bash
    ollama pull gemma3:4b
    ```

## How to Run the Pipeline

The project is run by executing the scripts in the `src/` directory in a specific order. For this example, we assume you have placed your PDF files in a folder named `documents/pdfs/`.

The `folder` argument in the scripts refers to this subfolder name (`pdfs`).

**Step 1: Place Your Documents**
-   Create a folder `documents/pdfs`.
-   Copy your PDF files into `documents/pdfs`.

**Step 2: Run OCR Processing**
This reads PDFs from `documents/pdfs`, extracts text, and saves it to `ocr_results/pdfs/`.
```bash
python src/ocr_processor.py pdfs
```

**Step 3: Clean the Extracted Text**
This step uses an LLM to clean each text file produced by the OCR.
```bash
# This command iterates through all generated .txt files and cleans them one by one.
for f in ocr_results/pdfs/*.txt; do python src/text_cleaner.py "pdfs/$(basename "$f")"; done
```

**Step 4: Chunk the Clean Documents**
This takes the cleaned text files and splits them into smaller chunks.
```bash
python src/doc_chunker.py pdfs
```

**Step 5: Generate Embeddings**
This creates vector embeddings for each chunk.
```bash
python src/embedding_generator.py pdfs
```

**Step 6: Populate the Vector Database**
This loads the embeddings and their metadata into the ChromaDB database.
```bash
python src/database.py pdfs
```

## How to Chat with Your Documents

After running the entire pipeline, you can ask questions using `chatbot.py`.

**Usage:**
```bash
python src/chatbot.py <folder_name> "<your_question>" [--provider <provider_name>]
```

**Examples:**

-   **Using OpenRouter (default):**
    ```bash
    python src/chatbot.py pdfs "Sebutkan para pihak yang terlibat dalam perjanjian?"
    ```

-   **Using Google Gemini:**
    ```bash
    python src/chatbot.py pdfs "What is the main objective of the agreement?" --provider google
    ```

-   **Using a local Ollama model:**
    ```bash
    python src/chatbot.py pdfs "Kapan perjanjian ini akan berakhir?" --provider ollama
    ```
