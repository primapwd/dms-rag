import os
from pathlib import Path
from dotenv import load_dotenv
import time
import argparse

from pdf2image import convert_from_path
import pytesseract

load_dotenv()

OCR_RESULTS_DIR = os.getenv("OCR_RESULTS_DIR", "processed_texts");
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents");

class OCRProcessor:
    """
    Sebuah kelas untuk memproses dokumen PDF, menjalankan OCR, dan menyimpan hasilnya.
    Didesain agar mudah dikonfigurasi dan dijalankan dari CLI.
    """
    def __init__(self, tesseract_path=None, poppler_path=None):
        """
        Inisialisasi OCR Processor.

        Args:
            tesseract_path (str, optional): Path eksplisit ke tesseract.exe (untuk Windows).
            poppler_path (str, optional): Path eksplisit ke folder bin Poppler (untuk Windows).
        """
        print("Menginisialisasi OCR Processor...")
        self.tesseract_path = tesseract_path
        self.poppler_path = poppler_path
        self._setup_dependencies()

    def _setup_dependencies(self):
        """Mengkonfigurasi path untuk Tesseract jika disediakan."""
        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            print(f"  -> Path Tesseract diatur ke: {self.tesseract_path}")

    def process_folder(self, input_folder_path, output_base_path):
        """
        Membaca semua file PDF dalam folder, menjalankan OCR, dan menyimpan teksnya.

        Args:
            input_folder_path (Path): Path ke folder input berisi file PDF.
            output_base_path (Path): Path dasar untuk menyimpan hasil .txt.

        Returns:
            list: Daftar dictionary berisi informasi dokumen yang diproses.
        """
        documents = []
        
        print(f"\nMulai memproses file dari folder: {input_folder_path.resolve()} dengan mode OCR")

        if not input_folder_path.is_dir():
            print(f"Error: Folder '{input_folder_path}' tidak ditemukan.")
            return []

        pdf_files = list(input_folder_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"Tidak ada file PDF yang ditemukan di folder '{input_folder_path}'.")
            return []

        print(f"Ditemukan {len(pdf_files)} file PDF untuk diproses.")
        
        output_folder = output_base_path / input_folder_path.name
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Hasil .txt akan disimpan di: {output_folder.resolve()}")

        for pdf_path in pdf_files:
            print(f"\n  -> Memproses file: {pdf_path.name}...")
            processed_doc = self._process_single_pdf(pdf_path, output_folder)
            if processed_doc:
                documents.append(processed_doc)

        return documents

    def _process_single_pdf(self, pdf_path, output_folder):
        """Memproses satu file PDF, menjalankan OCR, dan menyimpan hasilnya."""
        start_time = time.time()
        try:
            # Konversi PDF to gambar
            images = convert_from_path(pdf_path, poppler_path=self.poppler_path)
            
            full_text = ""
            # OCR Process
            for i, image in enumerate(images):
                print(f"     - OCR Halaman {i+1}/{len(images)}...")
                text = pytesseract.image_to_string(image, lang='ind+eng')
                full_text += text + "\n\n--- Akhir Halaman ---\n\n"
            
            # 3. Simpan hasil ke file .txt
            output_txt_path = output_folder / pdf_path.with_suffix('.txt').name
            output_txt_path.write_text(full_text, encoding='utf-8')
            
            end_time = time.time()
            print(f"     ... Selesai dalam {end_time - start_time:.2f} detik. Disimpan ke {output_txt_path.name}")
            
            return {
                "file_name": pdf_path.name,
                "content": full_text,
                "source_path": str(pdf_path),
                "processed_txt_path": str(output_txt_path)
            }
        except Exception as e:
            print(f"     Gagal memproses file {pdf_path.name}: {e}")
            return None


def main():
    """Fungsi utama untuk menjalankan skrip dari baris perintah."""
    parser = argparse.ArgumentParser(description="OCR PDF documents and save their content as .txt files.")
    parser.add_argument("folder", nargs='?', default="mous", help="Sub-folder di dalam direktori 'documents' (default: mous).")
    
    # Argumen opsional untuk path Tesseract dan Poppler (khusus Windows)
    parser.add_argument("--tesseract_path", help="Path eksplisit ke tesseract.exe.")
    parser.add_argument("--poppler_path", help="Path eksplisit ke folder bin Poppler.")
    
    args = parser.parse_args()

    # Setup path direktori
    SCRIPT_DIR = Path(__file__).parent
    BASE_DIR = SCRIPT_DIR.parent
    FOLDER_DOKUMEN = BASE_DIR / DOCUMENTS_DIR / args.folder
    FOLDER_OUTPUT = BASE_DIR / OCR_RESULTS_DIR
    
    processor = OCRProcessor(
        tesseract_path=args.tesseract_path,
        poppler_path=args.poppler_path
    )
    
    all_loaded_documents = processor.process_folder(FOLDER_DOKUMEN, FOLDER_OUTPUT)
    
    print("\n" + "="*50)
    print(f"Total dokumen yang berhasil di-OCR dan disimpan: {len(all_loaded_documents)}")
    print("="*50 + "\n")

    if all_loaded_documents:
        first_doc = all_loaded_documents[0]
        print(f"Contoh hasil dari dokumen pertama ('{first_doc['file_name']}'):")
        print(f"  - Preview Konten (500 karakter):\n")
        print(first_doc['content'][:100] + "...")


if __name__ == "__main__":
    main()