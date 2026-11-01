"""
Sesi 1 - Live Coding: Uji Coba Gemini API
==========================================
Versi MENTOR (kode lengkap dan berjalan)

Cara menjalankan:
    (venv) $ python mentor_test_gemini.py
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Muat variabel dari file .env (harus berisi GEMINI_API_KEY=...)
load_dotenv()

# Ambil API key dari environment
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    # Keep error/debugging messages in English (standard convention),
    # even though the surrounding comments are in Bahasa Indonesia.
    raise ValueError(
        "GEMINI_API_KEY not found. Make sure a .env file exists in the "
        "same folder and contains GEMINI_API_KEY=<your_key>."
    )

# Konfigurasi Gemini dengan API key
genai.configure(api_key=api_key)

# Inisialisasi model (gemini-1.5-flash: cepat dan hemat kuota untuk latihan)
model = genai.GenerativeModel("gemini-3.8-flash")

pertanyaan = input('mau tanya apa ke gemini: ')

# Kirim satu prompt sederhana
response = model.generate_content(pertanyaan)

# Tampilkan balasan
print("Gemini:", response.text)
