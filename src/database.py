import os
import chromadb
import argparse
import numpy as np
import json
import uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    """
    Sebuah kelas untuk mengelola ChromaDB, termasuk setup, memasukkan data,
    dan melakukan pencarian.
    """
    def __init__(self, db_path):
        """
        Inisialisasi Database Manager dengan path ke direktori database.

        Args:
            db_path (Path): Path ke folder tempat database akan disimpan.
        """
        print("Menginisialisasi Database Manager...")
        db_path.mkdir(exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(db_path))
        print(f"  -> Client ChromaDB terhubung ke direktori: {db_path.resolve()}")

    def setup_collection(self, collection_name, embeddings_path, metadata_path, force_recreate=True):
        """
        Menyiapkan sebuah koleksi, memasukkan data dari file, dan mengembalikannya.

        Args:
            collection_name (str): Nama koleksi yang akan dibuat.
            embeddings_path (Path): Path ke file .npy berisi vektor embeddings.
            metadata_path (Path): Path ke file .json berisi metadata chunk.
            force_recreate (bool): Jika True, koleksi lama dengan nama yang sama akan dihapus.

        Returns:
            chromadb.Collection: Objek koleksi yang sudah siap digunakan.
        """
        print(f"\n1. Menyiapkan koleksi '{collection_name}'...")
        
        if force_recreate and collection_name in [c.name for c in self.client.list_collections()]:
            print(f"   -> Menghapus koleksi lama '{collection_name}' sesuai permintaan.")
            self.client.delete_collection(name=collection_name)
        
        collection = self.client.get_or_create_collection(name=collection_name)
        
        if collection.count() == 0:
            print("   -> Koleksi kosong, memulai proses memasukkan data...")
            self._populate_collection(collection, embeddings_path, metadata_path)
        else:
            print("   -> Koleksi sudah berisi data, proses memasukkan data dilewati.")
            
        return collection

    def _populate_collection(self, collection, embeddings_path, metadata_path):
        """Metode internal untuk memasukkan data ke koleksi yang diberikan."""
        print("   -> Memuat data embedding dan metadata...")
        
        if not embeddings_path.exists() or not metadata_path.exists():
            print(f"   Error: File embedding atau metadata tidak ditemukan.")
            print(f"   - Cek Embedding: {embeddings_path}")
            print(f"   - Cek Metadata: {metadata_path}")
            return

        embeddings = np.load(embeddings_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        print(f"   -> Data dimuat. Jumlah embeddings: {len(embeddings)}, Jumlah metadata: {len(metadata)}")

        documents_to_add = [item['content'] for item in metadata]
        metadatas_to_add = [{'source': item['source_file']} for item in metadata]
        ids_to_add = [str(uuid.uuid4()) for _ in range(len(documents_to_add))]

        print("   -> Menambahkan data ke koleksi ChromaDB (dalam batch)...")
        batch_size = 100
        for i in range(0, len(documents_to_add), batch_size):
            collection.add(
                embeddings=embeddings[i:i+batch_size].tolist(),
                documents=documents_to_add[i:i+batch_size],
                metadatas=metadatas_to_add[i:i+batch_size],
                ids=ids_to_add[i:i+batch_size]
            )
            print(f"     - Menambahkan batch {i//batch_size + 1}...")

        print("   -> Proses penambahan data ke database selesai!")
        
    def perform_query(self, collection, query_text, model, n_results=3):
        """Melakukan pencarian pada koleksi yang diberikan."""
        print("\n" + "="*50)
        print(f"MELAKUKAN PENCARIAN")
        print(f"Query: '{query_text}'")
        print("="*50)

        query_embedding = model.encode(query_text).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        print(f"Hasil Pencarian ({n_results} Teratas):")
        if not results['documents'][0]:
            print("Tidak ada hasil yang ditemukan.")
        else:
            for i, doc in enumerate(results['documents'][0]):
                print(f"\n--- Hasil #{i+1} ---")
                print(f"Sumber: {results['metadatas'][0][i]['source']}")
                print(f"Relevansi (jarak): {results['distances'][0][i]:.4f}")
                print("Konten:")
                print(doc)
        
        print("\n" + "="*50)

def main():
    """Fungsi utama untuk menjalankan skrip dari baris perintah."""
    parser = argparse.ArgumentParser(description="Setup ChromaDB and optionally query it.")
    parser.add_argument("folder", nargs='?', default="mous", 
                        help="Prefix data yang akan diproses (default: mous).")
    parser.add_argument("--query", type=str, help="Teks query opsional untuk melakukan pencarian setelah setup.")
    
    args = parser.parse_args()

    EMBEDDED_DATA_DIR = os.getenv("EMBEDDINGS_DIR", "chunked_texts")

    # Setup path direktori
    BASE_DIR = Path(__file__).parent.parent
    EMBEDDED_DATA_DIR = BASE_DIR / EMBEDDED_DATA_DIR
    DB_DIR = BASE_DIR / 'db'
    
    EMBEDDINGS_FILE = EMBEDDED_DATA_DIR / f"{args.folder}_embeddings.npy"
    METADATA_FILE = EMBEDDED_DATA_DIR / f"{args.folder}_metadata.json"
    COLLECTION_NAME = args.folder

    db_manager = DatabaseManager(DB_DIR)

    collection = db_manager.setup_collection(
        collection_name=COLLECTION_NAME,
        embeddings_path=EMBEDDINGS_FILE,
        metadata_path=METADATA_FILE
    )
    
    if args.query:
        print("\nMemuat model untuk melakukan query...")
        model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
        K_RESULTS = int(os.getenv("K_RESULTS", 3))
        query_model = SentenceTransformer(model_name)
        db_manager.perform_query(collection, args.query, query_model, K_RESULTS)
    else:
        print("\nDatabase berhasil disiapkan. Untuk melakukan query, jalankan skrip lagi dengan argumen --query.")

if __name__ == "__main__":
    main()