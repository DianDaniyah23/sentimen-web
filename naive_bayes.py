import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Data training sederhana untuk memastikan aplikasi bisa berjalan
# Untuk hasil akurat, dataset ini harus jauh lebih besar dan berkualitas
TRAINING_DATA = {
    'teks': [
        'video ini sangat bagus dan menginspirasi', 'keren banget, suka!', 'penjelasan yang luar biasa', 'mantap',
        'konten sampah, tidak bermanfaat sama sekali', 'jelek, buang-buang waktu', 'saya tidak suka video ini', 'buruk',
        'lumayan lah', 'oke, cukup jelas', 'biasa saja', 'not bad'
    ],
    'sentimen': [
        'Positif', 'Positif', 'Positif', 'Positif',
        'Negatif', 'Negatif', 'Negatif', 'Negatif',
        'Netral', 'Netral', 'Netral', 'Netral'
    ]
}

# --- Model yang sudah "dilatih" ---
# Di aplikasi nyata, proses training ini dilakukan sekali dan modelnya disimpan (misal dengan joblib)
df = pd.DataFrame(TRAINING_DATA)
vectorizer = TfidfVectorizer(max_features=100)
X_train = vectorizer.fit_transform(df['teks'])
y_train = df['sentimen']

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
# --- Akhir dari proses training ---


def analyze_naive_bayes_sentiment(text):
    """
    Menganalisis sentimen teks menggunakan model Naive Bayes yang sudah dilatih.
    Mengembalikan label sentimen ('Positif', 'Negatif', 'Netral').
    
    Penting: Akurasi sangat bergantung pada data training. 
    Model ini hanya dilatih dengan data minimal.
    """
    if not text.strip():
        return 'Netral' # Kembalikan netral jika teks kosong

    # Ubah teks input menjadi format numerik yang sama dengan data training
    X_new = vectorizer.transform([text])
    
    # Lakukan prediksi
    prediction = nb_model.predict(X_new)[0]
    
    return prediction

