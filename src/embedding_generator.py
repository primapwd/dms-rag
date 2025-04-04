import os
import json
from pathlib import Path
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import time
from dotenv import load_dotenv

load_dotenv()

class EmbeddingGenerator:
    """
    Sebuah kelas untuk membuat embedding dari potongan teks (chunks)
    menggunakan model sentence-transformer.
    """
    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2'):
        """
        Inisialisasi Embedding Generator dan memuat model.

        Args:
            model_name (str): Nama model sentence-transformer dari Hugging Face Hub.
        """
        print("Menginisialisasi Embedding Generator...")
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Memuat model SentenceTransformer. Model akan diunduh jika belum ada."""
        print(f"  -> Memuat model embedding: '{self.model_name}'...")
        print("     Ini mungkin butuh waktu saat pertama kali dijalankan.")
        self.model = SentenceTransformer(self.model_name)
        print("  -> Model berhasil dimuat.")

    def generate(self, chunks_data):
        """
        Membuat embeddings dari daftar chunk yang diberikan.

        Args:
            chunks_data (list): List dari dictionary chunk, di mana setiap dict
                                memiliki key 'content'.

        Returns:
            numpy.ndarray: Sebuah array numpy berisi vektor embedding.
        """
        if not chunks_data:
            print("Peringatan: Tidak ada data chunk yang diberikan untuk di-embed.")
            return np.array([])

        texts_to_embed = [chunk['content'] for chunk in chunks_data]

        print(f"\nMulai membuat embeddings untuk {len(texts_to_embed)} chunks...")
        start_time = time.time()
        
        embeddings = self.model.encode(texts_to_embed, show_progress_bar=True)
        
        end_time = time.time()
        print(f"Proses embedding selesai dalam {end_time - start_time:.2f} detik.")
        
        return embeddings

def main():
    """Fungsi utama untuk menjalankan skrip dari baris perintah."""
    parser = argparse.ArgumentParser(description="Generate embeddings for chunked documents.")
    parser.add_argument("folder", nargs='?', default="mous", 
                        help="Prefix dari file chunk yang akan diproses (default: mous).")
    parser.add_argument("--model", default='paraphrase-multilingual-mpnet-base-v2',
                        help="Nama model sentence-transformer yang akan digunakan.")
    args = parser.parse_args()

    CHUNKED_RESULTS_DIR = os.getenv("CHUNKED_RESULTS_DIR", "chunked_texts")
    EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "chunked_texts")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")

    BASE_DIR = Path(__file__).parent.parent
    INPUT_FILE = BASE_DIR / CHUNKED_RESULTS_DIR / f"{args.folder}_chunks.json"
    OUTPUT_DIR = BASE_DIR / EMBEDDINGS_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"1. Memuat chunks dari: {INPUT_FILE}")
    if not INPUT_FILE.exists():
        print(f"Error: File chunk tidak ditemukan di '{INPUT_FILE}'.")
        print("Pastikan Anda sudah menjalankan skrip doc_chunker.py terlebih dahulu.")
        exit(1)
        
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunks_to_process = json.load(f)

    embedder = EmbeddingGenerator(model_name=EMBEDDING_MODEL)

    generated_embeddings = embedder.generate(chunks_to_process)
    
    if generated_embeddings.size > 0:
        embeddings_output_path = OUTPUT_DIR / f"{args.folder}_embeddings.npy"
        np.save(embeddings_output_path, generated_embeddings)
        
        metadata_output_path = OUTPUT_DIR / f"{args.folder}_metadata.json"
        with open(metadata_output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_to_process, f, indent=2, ensure_ascii=False)

        print("\n" + "="*50)
        print("Embedding berhasil dibuat dan disimpan!")
        print(f"  -> File Vektor (.npy): {embeddings_output_path}")
        print(f"  -> File Metadata (.json): {metadata_output_path}")
        print(f"\nBentuk (shape) dari array embeddings: {generated_embeddings.shape}")
        print(f"   (Terdapat {generated_embeddings.shape[0]} vektor, masing-masing dengan {generated_embeddings.shape[1]} dimensi)")
        print("="*50)

if __name__ == "__main__":
    main()