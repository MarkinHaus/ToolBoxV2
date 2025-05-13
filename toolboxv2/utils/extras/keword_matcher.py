import re
from collections import defaultdict


def calculate_keyword_score(text: str, keywords: set[str]) -> int:
    """
    Berechnet den Keyword-Score basierend auf der Häufigkeit der Keywords im Text.
    Case-insensitive und optimiert für Geschwindigkeit.

    :param text: Eingabetext als String
    :param keywords: Set von Keywords
    :return: Gesamt-Score als Integer
    """
    # Vorverarbeitung der Keywords
    keyword_pattern = re.compile(
        r'\b(' + '|'.join(re.escape(k.lower()) for k in keywords) + r')\b',
        flags=re.IGNORECASE
    )

    # Erstelle Frequenz-Wörterbuch
    freq_dict = defaultdict(int)

    # Finde alle Übereinstimmungen
    matches = keyword_pattern.findall(text.lower())

    # Zähle die Treffer
    for match in matches:
        freq_dict[match.lower()] += 1

    # Berechne Gesamt-Score
    total_score = sum(freq_dict.values())

    return total_score


def calculate_weighted_score(text: str, keyword_weights: dict or list) -> float:
    """
    Berechnet gewichteten Score mit unterschiedlichen Gewichten pro Keyword

    :param text: Eingabetext
    :param keyword_weights: Dictionary mit {Keyword: Gewicht}
    :return: Gewichteter Gesamt-Score
    """
    total = 0.0
    text_lower = text.lower()

    if isinstance(keyword_weights, list):
        keyword_weights = {k:v for k, v in keyword_weights}

    for keyword, weight in keyword_weights.items():
        count = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower))
        total += count * weight

    return round(total, 2)

STOPWORDS = {
        "der", "die", "das", "und", "oder", "in", "zu", "den",
        "ein", "eine", "von", "mit", "im", "am", "dem", "des",
    "a", "about", "above", "after", "again", "against", "all", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

def extract_keywords(
    text: str,
    max_len: int = -1,
    min_word_length: int = 3,
    with_weights: bool = False,
    remove_stopwords: bool = True,
    stopwords: bool = True
) -> list[str] | list[tuple[str, int]]:
    """
    Extrahiert Keywords mit optionaler Frequenzgewichtung

    :param text: Eingabetext
    :param max_len: Maximale Anzahl Keywords (-1 = alle)
    :param min_word_length: Minimale Wortlänge
    :param with_weights: Gibt Wort+Frequenz zurück wenn True
    :param remove_stopwords: Filtert deutsche Stopwörter
    :param german_stopwords: Verwendet deutsche Standard-Stopwörter
    :return: Keywords oder (Keyword, Häufigkeit) Paare
    """

    # Deutsche Basis-Stopwörter
    DEFAULT_STOPWORDS = STOPWORDS if stopwords else set()

    # Text vorverarbeiten
    words = re.findall(r'\b\w+\b', text.lower())

    # Worte filtern
    filtered_words = [
        word for word in words
        if len(word) > min_word_length
           and (not remove_stopwords or word not in DEFAULT_STOPWORDS)
    ]

    # Frequenzanalyse
    word_counts = defaultdict(int)
    for word in filtered_words:
        word_counts[word] += 1

    # Sortierung: Zuerst Häufigkeit, dann alphabetisch
    sorted_words = sorted(
        word_counts.items(),
        key=lambda x: (-x[1], x[0])
    )

    # Längenbegrenzung
    if max_len == -1:
        max_len = None
    result = sorted_words[:max_len]

    return result if with_weights else [word for word, _ in result]

# Beispielverwendung
if __name__ == "__main__":
    text = "Python ist eine großartige Sprache für Datenanalyse und Algorithmen. Python ermöglicht effiziente Algorithmen."
    keywords = {"Python", "Datenanalyse", "Algorithmus"}

    score = calculate_keyword_score(text, keywords)
    print(f"Keyword-Score: {score}")
    print(f"extract_keywords: {extract_keywords('Python ist Datenanalyse Algorithmus', remove_stopwords=True, with_weights=True)}")
    score = calculate_weighted_score(text, extract_keywords('Python ist Datenanalyse Algorithmus', remove_stopwords=True, with_weights=True))
    print(f"Keyword-Score: {score}")
    # Ausgabe: Keyword-Score: 3 (Python ×2, Datenanalyse ×1)
