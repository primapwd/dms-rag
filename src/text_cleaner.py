# src/cleansing.py

import os
import argparse
from pathlib import Path
import time
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class TextCleaner:
    """
    Sebuah kelas untuk membersihkan teks hasil OCR menggunakan berbagai LLM.
    Mendukung provider: 'google', 'openrouter', dan 'ollama'.
    """
    
    CLEANING_PROMPT_TEMPLATE = """
Anda adalah asisten ahli yang bertugas membersihkan teks yang diekstrak menggunakan OCR dari dokumen legal seperti Memorandum of Understanding (MoU). Tugas Anda adalah merapikan teks mentah di bawah ini.

Aturan Pembersihan:
1.  HAPUS semua elemen yang bukan bagian dari konten inti perjanjian, seperti header, footer, nomor halaman, logo, stempel, atau kop surat.
2.  PERTAHANKAN semua konten inti: judul, nomor surat, detail para pihak, semua pasal dan ayat, serta detail penandatanganan.
3.  PERBAIKI format teks agar mudah dibaca dengan normalisasi spasi dan baris baru.
4.  JANGAN menambahkan informasi apa pun yang tidak ada di teks asli.
5.  Output Anda HARUS HANYA berupa teks yang sudah bersih, tanpa penjelasan atau kalimat pembuka.

Teks Mentah Hasil OCR:
---
{raw_text}
---

Teks Bersih:
"""

    def __init__(self, provider, model_name, temperature=0.0):
        """
        Inisialisasi Text Cleaner berdasarkan provider yang dipilih.

        Args:
            provider (str): 'google', 'openrouter', atau 'ollama'.
            model_name (str): Nama model yang akan digunakan.
            temperature (float): Tingkat kreativitas model.
        """
        print(f"Menginisialisasi Text Cleaner dengan provider: '{provider.upper()}'...")
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature

        api_key = ''

        if self.provider == 'google':
            api_key = os.getenv("GOOGLE_API_KEY")

            if not api_key:
                raise ValueError("Untuk provider 'google', GOOGLE_API_KEY harus diatur.")
            genai.configure(api_key=api_key)
            self.llm_model = genai.GenerativeModel(self.model_name)
        
        elif self.provider == 'openrouter':
            api_key = os.getenv("OPENROUTER_API_KEY")

            if not api_key:
                raise ValueError("Untuk provider 'openrouter', OPENROUTER_API_KEY harus diatur.")
            self.llm_model = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

        elif self.provider == 'ollama':
            self.llm_model = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        
        else:
            raise ValueError(f"Provider '{self.provider}' tidak didukung. Pilih 'google', 'openrouter', atau 'ollama'.")

        print(f"  -> Siap menggunakan model: '{self.model_name}'")

    def clean_text(self, raw_text):
        """Membersihkan teks mentah dengan memanggil LLM yang sesuai."""
        prompt = self.CLEANING_PROMPT_TEMPLATE.format(raw_text=raw_text)
        
        try:
            if self.provider == 'google':
                response = self.llm_model.generate_content(prompt)
                return response.text.strip()
            
            elif self.provider in ['openrouter', 'ollama']:
                response = self.llm_model.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Anda adalah asisten ahli pembersih teks OCR untuk dokumen legal."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature
                )
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"  -> Error saat memanggil LLM ({self.provider}): {e}")
            return None

    def clean_file(self, file_path, output_path):
        """Membaca file, membersihkan kontennya, dan menyimpannya kembali."""
        print(f"\n1. Membaca file mentah: {file_path.name}")
        if not file_path.is_file():
            print(f"   Error: File tidak ditemukan di '{file_path}'")
            return False

        raw_text_content = file_path.read_text(encoding='utf-8')

        print(f"2. Mengirim teks ke '{self.provider.upper()}' untuk dibersihkan...")
        start_time = time.time()
        cleaned_content = self.clean_text(raw_text_content)
        end_time = time.time()

        if cleaned_content:
            print(f"   ... Selesai dalam {end_time - start_time:.2f} detik.")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(cleaned_content, encoding='utf-8')
            print(f"3. Sukses! File bersih telah disimpan di '{output_path.name}'.")
            return True
        else:
            print("3. Gagal membersihkan file.")
            return False

def main():
    """Fungsi utama untuk menjalankan skrip dari baris perintah."""
    parser = argparse.ArgumentParser(description="Clean a text file using a specified LLM provider.")
    parser.add_argument("input_file", type=str, help="Path ke file .txt yang akan dibersihkan.")
    args = parser.parse_args()

    llm_provider = os.getenv("LLM_PROVIDER", "google")
    model_name = os.getenv("CLEANSING_MODEL", "gemini-2.5-flash-lite")
    output_dir = os.getenv("CLEANSING_RESULTS_DIR", "cleaned_texts")
    ocr_output_path = os.getenv("OCR_RESULTS_DIR", "ocr_results")

    cleaner = TextCleaner(
        provider=llm_provider,
        model_name=model_name
    )

    BASE_DIR = Path(__file__).parent.parent
    input_path = BASE_DIR / ocr_output_path / args.input_file
    output_path = BASE_DIR / output_dir / args.input_file

    cleaner.clean_file(input_path, output_path)

if __name__ == "__main__":
    main()