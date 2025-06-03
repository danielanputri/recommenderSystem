# Laporan Proyek Machine Learning - Daniela Natali Putri

## Project Overview

**Latar Belakang Proyek**

Seiring dengan perkembangan dunia digital saat ini, banyak sekali informasi dan hiburan yang beredar. Salah satunya adalah film. Dengan begitu banyaknya film yang diproduksi, calon penonton sering kali mengalami kesulitan dalam menentukan film mana yang ingin ditonton. Proses pencarian film bisa memakan waktu, dan film yang akhirnya dipilih belum tentu sesuai dengan selera penonton setelah ditonton. Akibatnya, waktu yang dihabiskan menjadi lebih banyak. Selain itu, menonton film melalui bioskop, platform streaming, atau media fisik seperti DVD juga memerlukan biaya. Jika film yang ditonton ternyata tidak memuaskan, waktu dan biaya yang dikeluarkan pun akan terbuang sia-sia [1].

Mereka yang kesulitan memilih film untuk ditonton sering kali mencari di aplikasi twitter atau mengunjungi situs seperti suggestmemovie.com yang memberikan rekomendasi film kepada pengguna. Namun, dari berbagai solusi tersebut, banyak pengguna mengaku masih harus mencoba beberapa kali sebelum menemukan film yang dianggap bagus.

Sistem rekomendasi adalah alat untuk berinteraksi dengan ruang informasi yang besar dan kompleks [2]. Oleh karena itu, penting untuk memiliki sistem rekomendasi yang dapat menyederhanakan proses ini dengan memberikan rekomendasi yang relevan berdasarkan kebutuhan dan preferensi pengguna. Dengan memanfaatkan data rating pengguna sebelumnya, sistem rekomendasi ini diharapkan dapat membantu pengguna menemukan film yang paling sesuai dengan kebutuhan mereka.

**Pentingnya Proyek**

Proyek ini penting karena: 
- Peningkatan Pengalaman Pengguna: Membantu pengguna menemukan film yang sesuai dengan preferensi mereka, meningkatkan kepuasan dan pengalaman pengguna.
- Efisiensi: Mengurangi waktu dan usaha yang dibutuhkan pengguna dalam mencari film.
- Personalisasi: Memberikan rekomendasi yang dipersonalisasi berdasarkan data rating pengguna.

---
## Business Understanding

### Problem Statements
- Bagaimana kita bisa membantu pengguna mendapatkan film dengan genre paling sesuai dengan preferensi mereka?
- Bagaimana kita bisa membantu pengguna menemukan film yang mirip dengan film yang pernah dirating sebelumnya?

### Goals
Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Mengembangkan sistem rekomendasi yang dapat memberikan daftar film terbaik berdasarkan genre atau tema dari film yang disukai.
- Membangun sistem rekomendasi yang dapat memberikan daftar film terbaik berdasarkan film yang pernah dirating sebelumnya.

### Solution statements
- Menggunakan teknik **Content-Based Filtering** untuk mendapatkan film dengan genre yang mirip dengan film yang disukai.
- Menggunakan teknik **Collaborative Filtering** untuk mendapatkan rekomendasi film berdasarkan film yang pernah dirating.

## Data Understanding
Dataset ini dapat diunduh dari [kaggle](https://www.kaggle.com/datasets/aigamer/movie-lens-dataset). Dataset terbagi menjadi 4 yaitu links, movies, ratings, dan tags. Namun disini hanya menggunakan dataset movies dan ratings saja.

### Info Data movies.csv

|   # | Column   | Non-Null Count | Dtype   |
| --: | -------- | -------------- | ------- |
|   0 | movieId | 9742 non-null | int64   |
|   1 | title     | 9742 non-null | object  |
|   2 | genres    | 9742 non-null | object  |

Seperti yang terlihat di kolom **"movieId"** terdapat sampel sejumlah 9742, karena kolom **"movieId"** berisi id yang unik maka dapat disimpulkan bahwa panjang dataframe **movies.csv** adalah **9742**

Selanjutnya lihat contoh data dalam **movies.csv**
| movieId | title                   | genres                                    |
|---------|-------------------------|-------------------------------------------|
| 1       | Toy Story (1995)        | Adventure|Animation|Children|Comedy|Fantasy|
| 2       | Jumanji (1995)          | Adventure|Children|Fantasy              |
| 3       | Grumpier Old Men (1995) | Comedy|Romance                           |

Dalam kolom **"genres",** menggunakan pipe (|) sebagai pemisah. Hal ini akan membuat model machine learning lebih sulit dalam mengidentifikasi masing masing genre yang terdapat dalam suatu movie atau film.

Karena hal tersebut, mengetahui genre apa saja yang ada dalam dataframe tersebut akan sulit. Dataframe ini perlu dibersihkan terlebih dahulu di tahap [Data Preparation](#data-preparation)

### Info Data rating.csv

Dataset ini mengandung 100836 entri dan 3 kolom

|   # | Column   | Dtype |
| --: | -------- | ----- |
|   0 | userId  | int64 |
|   1 | movieId | int64 |
|   2 | rating   | int64 |
|   3 | timestamp | int64 |


Variabel-variabel pada dataset adalah sebagai berikut:
- movies:
  - `movieId`: ID unik untuk setiap film.
  - `title`: Judul film.
  - `genres`: Genre dari film.

- ratings:
  - `userId`: ID unik untuk setiap pengguna.
  - `movieId`: ID unik untuk setiap film (mengacu pada movies.csv).
  - `rating`: Rating yang diberikan pengguna untuk ponsel tertentu (skala 0.5 - 5.0).
  - `timestamp`: Menunjukkan detik sejak tengah malam Waktu Universal Terkoordinasi (UTC) tanggal 1 Januari 1970.

---
**Exploratory Data Analysis (EDA)**
- Rating Plot
  ![Rating Plot]()
  - Rating Paling Umum adalah 4.0: Frekuensi tertinggi ada pada skor rating 4.0.
  - Rating Tinggi Mendominasi: Skor rating 3.0, 3.5, 4.0, 4.5, dan 5.0 secara kolektif memiliki frekuensi yang jauh lebih tinggi dibandingkan skor rendah.
  - Skor di bawah 2.5 (terutama 0.5, 1.0, 1.5) memiliki frekuensi yang sangat rendah, mengindikasikan pengguna jarang memberikan penilaian sangat negatif.
    
- Movie User Plot
  ![Movie Plot]()
  - Terdapat perbedaan yang sangat signifikan antara jumlah film (sekitar 9600-9800) dengan jumlah pengguna (sekitar 600-700) dalam dataset.
  - Plot menunjukkan ketidakseimbangan yang besar, di mana jumlah item (film) yang tersedia jauh mendominasi jumlah pengguna yang ada.

---
## Data Preparation
Tahap data preparation dipisah menjai 2 yaitu untuk dataframe movie dan dataframe rating.

### Movie Data Preparation
**Teknik Data Preparation**
- Konversi genre dari setiap movie menjadi list.
- Cek missing value pada dataframe
- Cek genre unik
- Menghapus rows genre yang tidak diperlukan.
- Mengubah list genre menjadi string.

#### 1. Konversi genre dari setiap movie menjadi list.
- Pada tahap ini genre dari setiap movie pada dataframe **movies.csv** akan diubah menjadi bentuk array (list) dan menghapus pipe (|). Hal ini dilakukan untuk mempermudah akses ke genre di kolom "genre". Hasilnya sebagai berikut
| movieId | title                   | genres                                           | genre_str                            |
|---------|-------------------------|--------------------------------------------------|--------------------------------------|
| 0       | Toy Story (1995)        | [Adventure, Animation, Children, Comedy, Fantasy]  | Adventure Animation Children Comedy Fantasy |
| 1       | Jumanji (1995)          | [Adventure, Children, Fantasy]                   | Adventure Children Fantasy           |
| 2       | Grumpier Old Men (1995) | [Comedy, Romance]                                | Comedy Romance                       |

#### 2. Cek missing value pada dataframe.
- Pada tahap ini tidak ada missing value pada dataframe.

#### 3. Cek genre unik.
- Tahap ini seharusnya dilakukan di bagian [Data Understanding](#data-understanding) namun karena kedua tahap diatas perlu dijalankan terlebih dahulu sebelum bisa mengidentifikasi tiap genre yang terdapat dalam dataframe movie.
- Genre yang didapat adalah:
```
Total # of genre:  20
List of all genre availabel:  ['Adventure' 'Animation' 'Children' 'Comedy' 'Fantasy' 'Romance' 'Drama'
 'Action' 'Crime' 'Thriller' 'Horror' 'Mystery' 'Sci-Fi' 'War' 'Musical'
 'Documentary' 'IMAX' 'Western' 'Film-Noir' '(no genres listed)']
```
#### 4. Drop baris yang tidak digunakan.
- Pada tahap ini baris (no genres listed) akan dihapus karena tidak memiliki fitur konten yang bisa dibandingkan dengan film lain yang mana idak berguna dalam pendekatan **Content-Based Filtering**.

#### 5. Mengubah list genre menjadi string.
- Pada tahap ini list genre diubah lagi menjadi string dengan spasi sebagai pemisah dan dimasukkan ke kolom baru, sudah dilakukan pencegahan untuk genre 2 kata seperti "Toy Story" ketika diubah menjadi string maka spasi akan dihilangkan menjadi "ToyStory". Hal ini dilakukan untuk mempermudah **TF-IDF Vectorizer** dalam mendapatkan fitur genre.

### Rating Data Preparation
**Teknik Data Preparation**
- Cek missing value.
- Encoding userId dan movieId.
- Split menjadi data train dan validasi.

#### 1. Cek missing value pada dataframe.
- Pada tahap ini tidak ada missing value pada dataframe.

#### 2. Encoding userId dan movieId.
- Pada tahap ini dilakukan proses encoding pada kolom userId dan movieId dan dimasukkan ke kolom baru masing-masing. Hal ini dilakukan untuk merepresentasikan id user dan movie dalam format yang dapat di proses oleh model machine learning.

#### 3. Split menjadi data train dan validasi.
- Melakukan pemisahan pada dataframe menjadi train dan validasi dengan rasio 80:20, namun data di acak terlebih dahulu sebelum di pisah. Hal ini dilakukan supaya model dapat melakukan evaluasi pada data baru dan mencegah overfitting

---
## Modeling
Pada tahap ini akan membahas dua pendekatan utama yang digunakan dalam membangun sistem rekomendasi: Content-Based Filtering dan Collaborative Filtering. Berikut adalah penjelasan lebih lanjut mengenai parameter yang digunakan, kelebihan, dan kekurangan dari masing-masing pendekatan, serta beberapa potongan kode yang relevan.

**Model Sistem Rekomendasi Content Based Filtering**

Content-Based Filtering menggunakan deskripsi dan fitur dari item itu sendiri untuk memberikan rekomendasi. Berikut adalah parameter untuk pendekatan ini.

Parameter yang Digunakan:
  - TF-IDF Vectorizer: Untuk mengubah deskripsi teks menjadi vektor numerik.
  - Cosine Similarity: Untuk menghitung kesamaan antara vektor item.

Formula untuk **Cosine Similarity** adalah:  
$\displaystyle cos~(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$

Teknik ini menggunakan model **TF-IDF Vectorizer** untuk mendapatkan informasi mengenai genre yang terdapat di setiap film dan diubah menjadi fitur yang dapat diukur kemiripannya. Contohnya adalah sebagai berikut
| title                     | adventure | animation | drama    | film | musical | fi  | fantasy | imax | horror | sci |
|---------------------------|-----------|-----------|----------|------|---------|-----|---------|------|--------|-----|
| Hudsucker Proxy, The (1994) | 0.0       | 0.0       | 0.000000 | 0.0  | 0.0     | 0.0 | 0.0     | 0.0  | 0.0    | 0.0 |
| Semi-Pro (2008)            | 0.0       | 0.0       | 0.000000 | 0.0  | 0.0     | 0.0 | 0.0     | 0.0  | 0.0    | 0.0 |
| Hope and Glory (1987)      | 0.0       | 0.0       | 1.000000 | 0.0  | 0.0     | 0.0 | 0.0     | 0.0  | 0.0    | 0.0 |

Pada baris dan kolom yang memiliki angka lebih dari 0 menunjukan genre yang ada pada anime tersebut.

Setelah itu **Cosine Similarity** akan diterapkan pada dataframe anime yang telah dibersihkan sehingga menghasilkan output sebagai berikut:
| title                             | Bang, Bang, You're Dead (2002) | Educating Rita (1983) | He's Just Not That Into You (2009) | Air Up There, The (1994) | Deadly Outlaw: Rekka (2002) | Extreme Days (2001) | 17 Again (2009) | I Am David (2003) | Men, Women & Children (2014) | What Men Still Talk About (2011) |
|-----------------------------------|-------------------------------|------------------------|------------------------------------|---------------------------|-----------------------------|----------------------|------------------|-------------------|-------------------------------|----------------------------------|
| Eddie Izzard: Dress to Kill (1999) | 0.000000                      | 0.734682               | 0.504886                           | 1.000000                  | 0.000000                    | 0.402996             | 0.734682         | 0.000000          | 0.734682                      | 1.000000                         |
| Toy Story (1995)                 | 0.000000                      | 0.196445               | 0.135000                           | 0.267388                  | 0.000000                    | 0.369588             | 0.196445         | 0.000000          | 0.196445                      | 0.267388                         |
| Color Purple, The (1985)         | 1.000000                      | 0.678412               | 0.466216                           | 0.000000                  | 0.405263                    | 0.372130             | 0.678412         | 1.000000          | 0.678412                      | 0.000000                         |

Di tabel tersebut dapat dilihat kecocokan dari 1 anime dengan yang lain. Nilai-nilai pada tabel tersebut merepresentasikan persentase kecocokan antara kedua anime tersebut.

Bagaimana Algoritma Bekerja:
- Content-Based Filtering menggunakan model dari item itu sendiri untuk memberikan rekomendasi. Algoritma ini bekerja dengan cara mengubah fitur deskriptif item (model) menjadi representasi numerik menggunakan TF-IDF Vectorizer. Kemudian, cosine similarity dihitung untuk menentukan seberapa mirip item-item tersebut berdasarkan vektor fitur mereka. Berdasarkan kemiripan ini, sistem dapat merekomendasikan item yang paling mirip dengan item yang sudah disukai pengguna.

**Top-N Recommendation Content Based Filtering**
Tabel tersebut adalah dataframe cosine similarity yang akan digunakan untuk mendapatkan top-N rekomendasi film. Dalam kasus ini akan dicoba mendapatkan top-10 rekomendasi film yang mirip dengan film **"Avengers: Infinity War - Part I (2018)"**. Outputnya sebagai berikut
Data untuk uji coba:
| # | title | genres |
|--:|:------------------------------:|:--------------------------------:|
| 0 | Avengers: Infinity War - Part I (2018). | [Action, Adventure, Sci-Fi] |

Hasil rekomendasi:
| title                                               | genres                        |
|-----------------------------------------------------|-------------------------------|
| Rocketeer, The (1991)                               | [Action, Adventure, Sci-Fi]  |
| Ant-Man (2015)                                      | [Action, Adventure, Sci-Fi]  |
| Time Machine, The (2002)                            | [Action, Adventure, Sci-Fi]  |
| Iron Man (2008)                                     | [Action, Adventure, Sci-Fi]  |
| Sky Captain and the World of Tomorrow (2004)        | [Action, Adventure, Sci-Fi]  |
| Star Wars: Episode VI - Return of the Jedi (1983)   | [Action, Adventure, Sci-Fi]  |
| Justice League (2017)                               | [Action, Adventure, Sci-Fi]  |
| Farscape: The Peacekeeper Wars (2004)               | [Action, Adventure, Sci-Fi]  |
| Power/Rangers (2015)                                | [Action, Adventure, Sci-Fi]  |
| Black Panther (2017)                                | [Action, Adventure, Sci-Fi]  |

Berdasarkan hasil rekomendasi tersebut dapat dilihat bahwa film yang direkomendasikan memiliki genre yang mirip dengan input filmnya.

#### Kelebihan dan Kekurangan Content-Based Filtering

- Kelebihan:
  1. Tidak Perlu Data Pengguna Lain: Rekomendasi untuk seorang pengguna tidak bergantung pada data atau preferensi pengguna lain, hanya berdasarkan profil dan item yang disukainya sendiri.
  2. Mampu Merekomendasikan Item Baru: Selama item baru memiliki deskripsi fitur yang memadai, sistem dapat langsung merekomendasikannya tanpa perlu riwayat interaksi dari pengguna lain.
  3. Transparansi Rekomendasi: Alasan mengapa suatu item direkomendasikan bisa lebih mudah dijelaskan (misalnya, "karena Anda menyukai genre X, dan item ini juga bergenre X").

- Kekurangan:
  1. Keterbatasan Penemuan Hal Baru (Serendipity): Cenderung merekomendasikan item yang sangat mirip dengan yang sudah disukai pengguna, sehingga sulit menemukan variasi atau hal yang benar-benar baru.
  2. Sangat Bergantung pada Kualitas Fitur Item: Kualitas rekomendasi sangat dipengaruhi oleh seberapa baik dan lengkap fitur-fitur item diekstraksi dan direpresentasikan.
  3. Sulit Menangani Pengguna Baru (Cold Start User): Sistem kesulitan memberikan rekomendasi yang akurat untuk pengguna baru karena belum ada riwayat preferensi item untuk membangun profil pengguna.


**Model Sistem Rekomendasi Collaborative Filtering**

Collaborative Filtering menggunakan interaksi pengguna-item (rating) untuk memberikan rekomendasi. Berikut adalah parameter untuk pendekatan ini.

Proyek ini menggunakan model **RecommenderNet** yang dibuat dari kelas **Model** milik **Keras**.
Beberapa parameter yang Digunakan:
  - loss = MeanSquaredError
  - optimizer = Adam
  - learning_rate = 0.001
  - metrics = [RootMeanSquaredError (rmse), MeanAbsoluteError (mae)]

Parameter Callbacks:
1. EarlyStoppingg
   - monitor = 'val_loss'
   - patience = 5
   - restore_best_weights = True
2. ModelCheckpoint
   - filepath = 'best_model.keras'
   - monitor = 'val_loss'
   - save_best_only = True
3. ReduceLROnPlateau
   - monitor = 'val_loss'
   - factor = 0.5
   - patience = 3
   - min_lr = 1e-6

Bagaimana Algoritma Bekerja:
- Collaborative Filtering menggunakan interaksi pengguna-item (rating) untuk memberikan rekomendasi. Algoritma ini bekerja dengan cara memprediksi rating item yang belum diulas pengguna berdasarkan rating item yang mirip oleh pengguna lain. Model ini mempelajari pola preferensi pengguna dari data rating yang ada dan menggunakan pola tersebut untuk merekomendasikan item yang mungkin disukai pengguna.

**Top-N Recommendation Collaborative Filtering**
Pertama ambil dulu user secara acak dari dataframe rating. Kemudian tunjukkan film yang telah dirating oleh user tersebut.

```
Showing recommendations for user: 140
========================================
Movie with high ratings from user
----------------------------------------
Fly Away Home (1996) : Adventure, Children
Sound of Music, The (1965) : Musical, Romance
Great Escape, The (1963) : Action, Adventure, Drama, War
Grosse Pointe Blank (1997) : Comedy, Crime, Romance
Master and Commander: The Far Side of the World (2003) : Adventure, Drama, War
----------------------------------------
```
Kemudian akan diambil semua film yang belum dilihat oleh user, lalu model akan melakukan prediksi berdasarkan film dengan rating tinggi oleh user dan kemiripannya dengan user lain. Hasilnya akan mendapatkan rekomendasi sebagai berikut

```
Top 10 movie recommendations
----------------------------------------
Who Framed Roger Rabbit? (1988) : Adventure, Animation, Children, Comedy, Crime, Fantasy, Mystery
For Your Eyes Only (1981) : Action, Adventure, Thriller
Live and Let Die (1973) : Action, Adventure, Thriller
Thunderball (1965) : Action, Adventure, Thriller
Being John Malkovich (1999) : Comedy, Drama, Fantasy
Princess Mononoke (Mononoke-hime) (1997) : Action, Adventure, Animation, Drama, Fantasy
Insider, The (1999) : Drama, Thriller
Falling Down (1993) : Action, Drama
General, The (1926) : Comedy, War
Lost in Translation (2003) : Comedy, Drama, Romance
```

#### Kelebihan dan Kekurangan Content-Based Filtering

- Kelebihan:
  1. Menggunakan Data Pengguna: Memanfaatkan interaksi pengguna-item sehingga dapat merekomendasikan item yang tidak mirip tetapi disukai oleh pengguna dengan preferensi serupa.
  2. Menangani Data Besar: Dapat bekerja dengan data besar dan menemukan pola-pola kompleks.
 
- Kekurangan:
  1. Cold Start Problem: Kesulitan merekomendasikan item baru atau kepada pengguna baru yang belum memiliki cukup interaksi.

---
## Evaluation
Pada bagian ini, akan mengevaluasi model rekomendasi yang telah dibangun menggunakan metrik evaluasi yang tepat. Untuk model prediksi rating, kita akan menggunakan Mean Absolute Error (MAE) sebagai metrik evaluasi. Selain itu, akan mengevaluasi apakah proyek ini berhasil menjawab problem statement dan memberikan solusi yang diinginkan.

**Metrik Evaluasi**

MAE atau Mean Absolute Error diterapkan dengan cara mengukur rata-rata dari selisih absolut antara prediksi dan nilai asli (y_asli - y_prediksi).

MAE = $\displaystyle \sum\frac{|y_i - \hat{y}_i|}{n}$

Dimana:
MAE = nilai Mean Absolute Error
y = nilai aktual
ŷ = nilai prediksi
i = urutan data
n = jumlah data

Berikut plot MAE dari model:
![Grafik train vs test]()

**Evaluasi Terhadap Business Understanding**
- Menjawab Problem Statement: Model yang dibuat berhasil menjawab problem statement dengan memberikan rekomendasi film berdasarkan model yang ada. Pendekatan content-based filtering menggunakan model film untuk memberikan rekomendasi yang relevan berdasarkan genre, sementara collaborative filtering memanfaatkan interaksi pengguna-item (rating) sebelumnya untuk menemukan pola preferensi pengguna.

- Mencapai Goals: Model content-based filtering dengan cosine similarity dan collaborative filtering dengan RecommenderNet berhasil mencapai tujuan untuk memberikan rekomendasi film yang relevan.

---
## Kesimpulan
Dengan menggunakan kedua pendekatan ini, kita dapat membangun sistem rekomendasi yang lebih robust dan fleksibel. Content-Based Filtering cocok untuk memberikan rekomendasi berdasarkan fitur-fitur item itu sendiri, sementara Collaborative Filtering efektif dalam menemukan pola-pola preferensi pengguna dari data interaksi yang ada. Memahami kelebihan dan kekurangan masing-masing pendekatan membantu kita memilih metode yang paling sesuai dengan kebutuhan dan konteks spesifik dari sistem rekomendasi yang sedang dibangun. Namun proyek ini masih belum memberikan solusi untuk kasus **Cold Start**. Dimana user baru belum memiliki rating film maupun jenis film yang disukai.

---
## Referensi

[1] [Fajriansyah, M., Adikara, P. P., & Widodo, A. W. (2021). Sistem Rekomendasi Film Menggunakan Content Based Filtering. Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer, 5(6), 2188-2199.](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/9163)

[2] [R. Burke, A. Felfernig, and M. H. Göker, “Recommender Systems: An Overview”, AIMag, vol. 32, no. 3, pp. 13-18, Jun. 2011.](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2361) 
