# 🌿 Proyek Klasifikasi Tanaman 🌱

Selamat datang di proyek klasifikasi tanaman! 🌼 Proyek ini bertujuan membangun model deep learning untuk mengklasifikasikan gambar tanaman ke dalam kategori masing-masing menggunakan TensorFlow dan Keras. Dataset diambil dari Kaggle, dengan langkah-langkah pra-pemrosesan, augmentasi data, pelatihan, dan evaluasi model. 🚀

## 📑 Daftar Isi
- [Ikhtisar Proyek](#ikhtisar-proyek-📖)
- [Dataset](#dataset-📊)
- [Persyaratan](#persyaratan-🛠️)
- [Instalasi](#instalasi-⚙️)
- [Penggunaan](#penggunaan-🎮)
- [Arsitektur Model](#arsitektur-model-🧠)
- [Pelatihan dan Evaluasi](#pelatihan-dan-evaluasi-🏋️)
- [Hasil](#hasil-🏆)
- [Kontribusi](#kontribusi-🤝)
- [Kontak](#kontak-📧)

## Ikhtisar Proyek 📖
Proyek ini mengembangkan model jaringan saraf konvolusional (CNN) untuk mengklasifikasikan gambar tanaman. ✨ Kami menggunakan pembelajaran transfer dengan model seperti **MobileNet** dan **DenseNet121**, ditambah augmentasi data untuk performa maksimal. Proyek ini dijalankan di **Google Colab dengan T4 GPU** untuk pelatihan cepat! ⚡

## Dataset 📊
Dataset berasal dari Kaggle: [Plants Classification Dataset](https://www.kaggle.com/datasets/marquis03/plants-classification). 🖼️ Dataset berisi gambar tanaman yang dikelompokkan berdasarkan kelas, dibagi menjadi:
- **Data Train**
- **Data Val**
- **Data Test**

Untuk mengunduh, Anda perlu akun Kaggle dan kunci API. Notebook sudah menyertakan kode untuk mengunduh dan mengekstrak dataset. 📥

## Persyaratan 🛠️
Proyek ini menggunakan Python dan pustaka berikut, yang sudah tersedia atau dapat diinstal di Google Colab:
- TensorFlow 🤖
- Keras 🧠
- NumPy 🔢
- Pandas 📈
- Matplotlib 📊
- Seaborn 🎨
- Scikit-learn 📚
- OpenCV 🖼️
- Scikit-image 🖌️
- TQDM ⏳
- Pillow 🛏️
  
## Instal Dependensi
Jalankan perintah berikut di sel notebook untuk menginstal dependensi:
```bash
!pip install -r requirements.txt
```
Catatan: Jika Colab sudah memiliki beberapa pustaka (seperti TensorFlow atau NumPy), instalasi akan melewati pustaka tersebut. Pastikan semua pustaka terinstal dengan baik! ✅

## Instalasi ⚙️
Proyek ini dijalankan di **Google Colab dengan T4 GPU**, jadi tidak perlu lingkungan lokal. Ikuti langkah berikut:

1. **Buka Google Colab** 🌐
   - Unggah `Template_Submission_Akhir_(Dearmawan).ipynb` atau buat notebook baru.
   - Pastikan runtime menggunakan **T4 GPU**:
     - Klik `Runtime` > `Change runtime type` > Pilih `T4 GPU`. ✅

2. **Siapkan Kaggle API** 🔑
   - **Langkah 1: Buat Akun Kaggle**  
     Jika belum punya akun Kaggle, daftar di [kaggle.com](https://www.kaggle.com). 📝
   - **Langkah 2: Dapatkan Kunci API**  
     - Masuk ke akun Kaggle Anda. 🔐
     - Klik ikon profil di kanan atas, lalu pilih **Settings**. ⚙️  
     - Di bagian **API**, klik tombol **Create New Token**. 🔑  
     - File bernama `kaggle.json` akan otomatis terunduh ke folder **Downloads** di komputer/laptop Anda. 💾
   - **Langkah 3: Unggah File `kaggle.json` ke Colab**  
     - Di notebook Colab, jalankan kode berikut untuk mengunggah file:  
       ```python
       from google.colab import files
       files.upload()
       ```
     - Setelah itu, klik **Choose Files**, lalu pilih file `kaggle.json` dari folder **Downloads** di komputer/laptop Anda. 📤
   - **Langkah 4: Atur Kunci API**  
     Jalankan perintah berikut di sel notebook untuk mengatur kunci API:  
     ```bash
     !mkdir -p ~/.kaggle
     !cp kaggle.json ~/.kaggle/
     !chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Unduh Dataset** 📦
   Jalankan perintah berikut di notebook untuk mengunduh dan mengekstrak dataset:
   ```bash
   !kaggle datasets download -d marquis03/plants-classification
   !unzip plants-classification.zip
   ```

4. **Instal Pustaka Tambahan** (jika perlu) 📚
   Colab biasanya sudah memiliki TensorFlow dan NumPy. Jika diperlukan pustaka lain, tambahkan:
   ```bash
   !pip install opencv-python scikit-image tqdm
   ```

## Penggunaan 🎮
1. **Buka Notebook** 📓
   Buka `Template_Submission_Akhir_(Dearmawan).ipynb` di Google Colab.

2. **Jalankan Sel** ▶️
   - Eksekusi sel secara berurutan untuk:
     - Mengunduh dan memproses dataset. 📥
     - Melakukan augmentasi data. 🖌️
     - Membangun dan melatih model. 🏋️
     - Mengevaluasi performa model. 📊
   - Pastikan runtime menggunakan T4 GPU untuk pelatihan cepat! ⚡

3. **Ubah Parameter (Opsional)** ⚙️
   - Sesuaikan hiperparameter seperti learning rate, batch size, atau epoch di notebook.
   - Coba arsitektur model atau teknik augmentasi berbeda untuk eksperimen. 🧪
     
## Arsitektur Model 🧠
Proyek ini menggabungkan CNN kustom dan pembelajaran transfer:
- **CNN Kustom**: Menggunakan lapisan `Conv2D`, `MaxPooling2D`, `Dense`, `Dropout`, dan `BatchNormalization`. 🛠️
- **Transfer Learning**: Memanfaatkan `MobileNet` dan `DenseNet121` dengan lapisan tambahan untuk klasifikasi. 🚀
- **Augmentasi Data**: Rotasi, pembalikan, zoom, dan penyesuaian gamma untuk dataset lebih beragam. 🎨

Model dikompilasi dengan optimizer **Adam** dan fungsi kerugian **categorical cross-entropy** untuk klasifikasi multi-kelas. 📈

## Pelatihan dan Evaluasi 🏋️
- **Pelatihan**:
  - Dataset dibagi menjadi pelatihan dan validasi. 📊
  - Augmentasi data diterapkan untuk mencegah overfitting. 🛡️
  - Callback seperti `EarlyStopping`, `ModelCheckpoint`, dan `ReduceLROnPlateau` digunakan untuk pelatihan optimal. ⚙️
- **Evaluasi**:
  - Metrik: akurasi, kerugian, matriks kebingungan, dan laporan klasifikasi. 📊
  - Visualisasi: plot akurasi dan kerugian pelatihan/validasi dengan Matplotlib dan Seaborn. 📈

## Hasil 🏆
Notebook menyediakan kode untuk mengevaluasi model pada data validasi. Hasil utama:
- **Akurasi**: Persentase gambar yang diklasifikasikan benar. ✅
- **Matriks**: Distribusi prediksi di seluruh kelas. 📉
- **Laporan Klasifikasi**: Presisi, recall,dan F1-score per kelas. 📋

Jalankan sel evaluasi untuk melihat hasil. Model disimpan di direktori `models/` untuk penggunaan ulang. 💾

## Kontribusi 🤝
Kami menyambut kontribusi! 🎉 Untuk berkontribusi:
1. Fork repositori. 🍴
2. Buat branch baru (`git checkout -b fitur-baru`). 🌿
3. Lakukan perubahan dan commit (`git commit -m "Menambahkan fitur"`). 💻
4. Push ke branch (`git push origin fitur-baru`). 🚀
5. Buka pull request. 📬

Pastikan kode mengikuti gaya proyek dan sertakan dokumentasi. 📝

## Kontak 📧
- **Penulis**: Dearmawan
- **Email**: dearmawantan@gmail.com 📨

Ada pertanyaan atau saran? Hubungi via email atau buka isu di repositori. 😊

🌟 **Selamat mengklasifikasikan tanaman!** 🌟
