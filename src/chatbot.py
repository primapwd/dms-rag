# src/chatbot.py

import argparse
import chromadb
import os
import openai
import google.generativeai as genai

from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

class Chatbot:
    """
    Kelas Chatbot RAG yang mendukung provider Google, OpenRouter, dan Ollama.
    """
    PROMPT_TEMPLATE = """
    Berdasarkan kumpulan konteks di bawah ini, jawablah pertanyaan berikut.
    Jawablah dengan jelas, ringkas, dan hanya berdasarkan informasi dari konteks yang diberikan.
    Jika informasi tidak ditemukan dalam konteks, katakan "Maaf, informasi tersebut tidak ditemukan dalam dokumen saya."

    Kumpulan Konteks:
    ---
    {context}
    ---

    Pertanyaan:
    {query}
    """

    def __init__(self, collection_name, db_path, llm_provider='openrouter', llm_model_name=None):
        """
        Inisialisasi Chatbot.

        Args:
            collection_name (str): Nama koleksi di ChromaDB.
            db_path (str): Path ke direktori database.
            llm_provider (str): 'google', 'openrouter', atau 'ollama'.
            llm_model_name (str, optional): Nama model LLM yang akan digunakan.
        """
        print("Menginisialisasi Chatbot RAG...")
        self.provider = llm_provider

        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
        print(f"  -> Memuat model embedding: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        print(f"  -> Menghubungkan ke database di: {db_path}")
        db_client = chromadb.PersistentClient(path=str(db_path))
        self.collection = db_client.get_or_create_collection(name=collection_name)
        print(f"  -> Terhubung ke koleksi: '{collection_name}' (Total item: {self.collection.count()})")

        self._setup_llm(llm_model_name)

    def _setup_llm(self, llm_model_name=None):
        """Metode internal untuk menyiapkan koneksi ke LLM."""
        print(f"  -> Mengkonfigurasi LLM provider: {self.provider.upper()}")
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", 0.1))
        
        if self.provider == 'google':
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            if not self.google_api_key: raise ValueError("GOOGLE_API_KEY tidak ditemukan di .env")
            genai.configure(api_key=self.google_api_key)
            self.llm_model_name = llm_model_name or os.getenv("LLM_MODEL_GOOGLE", "gemini-1.5-flash-latest")
            self.llm_client = genai.GenerativeModel(self.llm_model_name)
        
        elif self.provider == 'openrouter':
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.openrouter_api_key: raise ValueError("OPENROUTER_API_KEY tidak ditemukan di .env")
            self.llm_model_name = llm_model_name or os.getenv("LLM_MODEL_OPENROUTER", "meta-llama/llama-3-8b-instruct")
            self.llm_client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.openrouter_api_key)

        elif self.provider == 'ollama':
            # API key tidak diperlukan, tapi base_url harus benar
            self.llm_model_name = llm_model_name or os.getenv("LLM_MODEL_OLLAMA", "gemma:2b") # Default model untuk Ollama
            self.llm_client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        
        else:
            raise ValueError(f"Provider '{self.provider}' tidak didukung.")
            
        print(f"  -> Siap menggunakan model LLM: '{self.llm_model_name}'")

    def _retrieve_context(self, query_text, n_results):
        """Mengambil konteks yang relevan dari database."""
        print(f"\n1. (Retrieval) Mencari konteks untuk query: '{query_text}'")
        query_embedding = self.embedding_model.encode(query_text)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        retrieved_docs = results['documents'][0]
        context_str = "\n".join(f"- {doc}" for doc in retrieved_docs)
        print(f"   -> Konteks ditemukan ({len(retrieved_docs)} chunks).")
        return context_str

    def _generate_answer(self, context, query):
        """Membuat jawaban menggunakan LLM berdasarkan konteks."""
        print(f"2. (Generation) Mengirim konteks dan query ke {self.provider.upper()}...")
        prompt = self.PROMPT_TEMPLATE.format(context=context, query=query)
        
        try:
            if self.provider == 'google':
                response = self.llm_client.generate_content(prompt)
                return response.text.strip()
            
            elif self.provider in ['openrouter', 'ollama']:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[
                        {"role": "system", "content": "Anda adalah asisten Q&A yang menjawab berdasarkan konteks."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature
                )
                return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Terjadi kesalahan saat memanggil API {self.provider.upper()}: {e}"

    def ask(self, query_text, k_results=5):
        """Metode utama untuk berinteraksi dengan chatbot."""
        context = self._retrieve_context(query_text, n_results=k_results)
        answer = self._generate_answer(context, query_text)
        return answer

def main():
    """Fungsi utama untuk menjalankan chatbot dari baris perintah."""
    parser = argparse.ArgumentParser(description="Tanya jawab dengan dokumen Anda menggunakan RAG.")
    parser.add_argument("folder", help="Nama koleksi/folder data yang akan digunakan (misal: mous).")
    parser.add_argument("query", help="Pertanyaan yang ingin Anda ajukan.")
    parser.add_argument(
        "--provider",
        default="openrouter",
        choices=['google', 'openrouter', 'ollama'], # Menambahkan 'ollama'
        help="Pilih LLM provider yang akan digunakan."
    )
    parser.add_argument("--k", type=int, default=5, help="Jumlah chunk konteks yang akan diambil.")
    
    args = parser.parse_args()

    BASE_DIR = Path(__file__).parent.parent
    DB_DIR = BASE_DIR / 'db'
    
    # Pilih nama model berdasarkan provider
    if args.provider == 'google':
        model_name = os.getenv("LLM_MODEL_GOOGLE", "gemini-1.5-flash-latest")
    elif args.provider == 'openrouter':
        model_name = os.getenv("LLM_MODEL_OPENROUTER", "meta-llama/llama-3-8b-instruct")
    else: # Ollama
        model_name = os.getenv("LLM_MODEL_OLLAMA", "gemma:2b")

    try:
        chatbot = Chatbot(
            collection_name=args.folder,
            db_path=DB_DIR,
            llm_provider=args.provider,
            llm_model_name=model_name
        )
    except Exception as e:
        print(f"Gagal menginisialisasi chatbot: {e}")
        return

    final_answer = chatbot.ask(args.query, k_results=args.k)

    print("\n✅ Jawaban Akhir dari Sistem RAG:")
    print("--------------------------------------------------")
    print(final_answer)
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()