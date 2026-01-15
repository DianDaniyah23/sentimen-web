from textblob import TextBlob

def analyze_textblob_sentiment(text):
    """
    Menganalisis sentimen teks menggunakan TextBlob.
    Mengembalikan label sentimen ('Positif', 'Negatif', 'Netral') dan skor polaritas.
    """
    analysis = TextBlob(text)
    # Skor polaritas berada di antara -1 (sangat negatif) dan 1 (sangat positif)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = 'Positif'
    elif polarity < -0.1:
        sentiment = 'Negatif'
    else:
        sentiment = 'Netral'
        
    return sentiment, round(polarity, 2)
