from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_vader_sentiment(text):
    """
    Menganalisis sentimen teks menggunakan VADER.
    Mengembalikan label sentimen ('Positif', 'Negatif', 'Netral') dan skor compound.
    """
    analyzer = SentimentIntensityAnalyzer()
    # Skor VADER memberikan 'compound' score dari -1 (paling negatif) hingga 1 (paling positif)
    vader_scores = analyzer.polarity_scores(text)
    compound_score = vader_scores['compound']
    
    if compound_score >= 0.05:
        sentiment = 'Positif'
    elif compound_score <= -0.05:
        sentiment = 'Negatif'
    else:
        sentiment = 'Netral'
        
    return sentiment, compound_score
