import os
from pathlib import Path
import argparse
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

class DocumentChunker:
    """
    Sebuah kelas untuk memecah dokumen teks menjadi potongan-potongan (chunks)
    yang lebih kecil dan menambahkan metadata.
    """
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Inisialisasi Document Chunker.

        Args:
            chunk_size (int): Ukuran maksimal setiap chunk dalam karakter.
            chunk_overlap (int): Jumlah karakter yang tumpang tindih antar chunk.
        """
        print("Menginisialisasi Document Chunker...")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        print(f"  -> Ukuran chunk: {chunk_size}, Tumpang tindih: {chunk_overlap}")

    def chunk_folder(self, input_folder_path):
        """
        Membaca semua file .txt dari folder, melakukan chunking, dan mengembalikan hasilnya.

        Args:
            input_folder_path (Path): Path ke folder berisi file .txt yang akan di-chunk.

        Returns:
            list: Sebuah list berisi dictionary untuk setiap chunk.
        """
        all_chunks = []
        print(f"\nMulai proses chunking dari folder: {input_folder_path.resolve()}")

        txt_files = list(input_folder_path.glob("*.txt"))
        if not txt_files:
            print(f"Tidak ada file .txt ditemukan di {input_folder_path}")
            return []

        print(f"Ditemukan {len(txt_files)} file untuk di-chunk.")
        
        for txt_path in txt_files:
            print(f"  -> Memproses file: {txt_path.name}")
            
            # Baca konten teks
            document_content = txt_path.read_text(encoding='utf-8')
            
            # Lakukan proses chunking
            chunks = self.text_splitter.split_text(document_content)
            
            # Tambahkan metadata ke setiap chunk
            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "source_file": txt_path.name,
                    "chunk_sequence_id": i + 1,
                    "content": chunk_text
                })
            
            print(f"     ... menghasilkan {len(chunks)} chunks.")

        print(f"\nTotal chunk yang berhasil dibuat dari semua file: {len(all_chunks)}")
        return all_chunks


def main():
    """Fungsi utama untuk menjalankan skrip dari baris perintah."""
    parser = argparse.ArgumentParser(description="Chunk text documents from a folder.")
    parser.add_argument("folder", nargs='?', default="mous", 
                        help="Sub-folder di dalam direktori 'cleansed_texts' (default: mous).")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Ukuran maksimal setiap chunk.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Jumlah tumpang tindih antar chunk.")
    
    args = parser.parse_args()

    output_dir = os.getenv("CLEANSING_RESULTS_DIR", "cleaned_texts")
    chunked_dir = os.getenv("CHUNKED_RESULTS_DIR", "chunked_texts")

    BASE_DIR = Path(__file__).parent.parent
    
    FOLDER_INPUT = BASE_DIR / output_dir / args.folder
    FOLDER_OUTPUT = BASE_DIR / chunked_dir
    FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)

    chunker = DocumentChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    chunked_documents = chunker.chunk_folder(FOLDER_INPUT)
    
    if chunked_documents:
        output_file_path = FOLDER_OUTPUT / f"{args.folder}_chunks.json"
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(chunked_documents, f, indent=2, ensure_ascii=False)
        
        print(f"\nHasil chunking telah disimpan di: {output_file_path}")
        
        print("\n" + "="*50)
        print("Contoh 2 chunk pertama:")
        print(json.dumps(chunked_documents[:2], indent=2, ensure_ascii=False))
        print("="*50)

if __name__ == "__main__":
    main()