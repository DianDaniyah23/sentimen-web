import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    def __init__(self):
        # --- Dummy Dataset ---
        # Dataset ini digunakan untuk melatih model secara "on-the-fly".
        # Di aplikasi nyata, Anda akan memiliki dataset yang lebih besar dan disimpan terpisah.
        self.dummy_data = {
            'comment': [
                "Keren banget videonya!", "Sangat bermanfaat.", "Aku suka sekali.",
                "Tidak jelas.", "Membosankan.", "Konten sampah.",
                "Biasa saja.", "Cukup informatif.", "Lumayan lah."
            ],
            'sentiment': [
                'positif', 'positif', 'positif',
                'negatif', 'negatif', 'negatif',
                'netral', 'netral', 'netral'
            ]
        }
        self.df_train = pd.DataFrame(self.dummy_data)
        
        # Inisialisasi model
        self.vectorizer = TfidfVectorizer(preprocessor=self._preprocess_text)
        self.svm_model = SVC(kernel='linear')
        self.nb_model = MultinomialNB()
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # Membuat pipeline untuk setiap algoritma
        self.svm_pipeline = self._create_pipeline(self.svm_model)
        self.nb_pipeline = self._create_pipeline(self.nb_model)

    def _preprocess_text(self, text):
        """Fungsi untuk membersihkan teks."""
        text = text.lower()  # Ubah ke huruf kecil
        text = re.sub(r'\d+', '', text)  # Hapus angka
        text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
        text = text.strip()  # Hapus spasi di awal dan akhir
        return text

    def _create_pipeline(self, model):
        """Membuat pipeline scikit-learn."""
        return Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', model)
        ])

    def _train(self):
        """Melatih model SVM dan Naive Bayes."""
        X_train = self.df_train['comment']
        y_train_svm = self.df_train['sentiment']
        y_train_nb = self.df_train['sentiment']

        # Melatih pipeline
        self.svm_pipeline.fit(X_train, y_train_svm)
        self.nb_pipeline.fit(X_train, y_train_nb)

    def analyze(self, comments_to_analyze, algorithm='svm'):
        """
        Menganalisis sentimen dari daftar komentar.
        
        Args:
            comments_to_analyze (list): Daftar string komentar yang akan dianalisis.
            algorithm (str): 'svm', 'naive_bayes', atau 'lexicon'.
            
        Returns:
            list: Daftar dictionary berisi 'comment' dan 'sentiment'.
        """
        results = []

        if algorithm == 'lexicon':
            for comment in comments_to_analyze:
                score = self.vader_analyzer.polarity_scores(comment)
                sentiment = 'Netral'
                if score['compound'] >= 0.05:
                    sentiment = 'Positif'
                elif score['compound'] <= -0.05:
                    sentiment = 'Negatif'
                results.append({'Komentar': comment, 'Sentimen': sentiment})
            return results

        # Latih model ML jika bukan lexicon
        self._train()

        if algorithm == 'svm':
            model = self.svm_pipeline
        elif algorithm == 'naive_bayes':
            model = self.nb_pipeline
        else:
            raise ValueError("Algoritma tidak valid. Pilih 'svm', 'naive_bayes', atau 'lexicon'.")

        # Lakukan prediksi
        predictions = model.predict(comments_to_analyze)
        
        # Gabungkan komentar dengan hasil prediksi
        for comment, sentiment in zip(comments_to_analyze, predictions):
            results.append({'Komentar': comment, 'Sentimen': sentiment.capitalize()})
            
        return results

# Contoh penggunaan (opsional, untuk pengujian)
if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    
    new_comments = [
        "Video ini luar biasa dan sangat membantu!",
        "Saya tidak mengerti apa-apa, penjelasannya buruk.",
        "Cukup ok, tidak ada yang istimewa."
    ]
    
    # Analisis dengan SVM
    print("--- Hasil Analisis SVM ---")
    svm_results = analyzer.analyze(new_comments, algorithm='svm')
    print(pd.DataFrame(svm_results))
    
    print("\n" + "="*30 + "\n")
    
    # Analisis dengan Naive Bayes
    print("--- Hasil Analisis Naive Bayes ---")
    nb_results = analyzer.analyze(new_comments, algorithm='naive_bayes')
    print(pd.DataFrame(nb_results))

    print("\n" + "="*30 + "\n")

    # Analisis dengan Lexicon (VADER)
    print("--- Hasil Analisis Lexicon-Based (VADER) ---")
    lexicon_results = analyzer.analyze(new_comments, algorithm='lexicon')
    print(pd.DataFrame(lexicon_results))