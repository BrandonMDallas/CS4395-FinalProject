import spacy
import re

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

def preprocess_text(text: str, remove_stopwords: bool = True, remove_punct: bool = True) -> str:
    """
    Preprocess text by lemmatizing, removing stopwords, and cleaning.
    
    Args:
        text (str): Input text to preprocess.
        remove_stopwords (bool): Whether to remove stopwords.
        remove_punct (bool): Whether to remove punctuation.
    
    Returns:
        str: Preprocessed text.
    """
    # Pre-clean text to handle numbers and special cases
    text = re.sub(r"(\d+)([a-zA-Z])", r"\1 \2", text)  # Separate numbers attached to letters
    text = re.sub(r"([a-zA-Z])(\d+)", r"\1 \2", text)  # Separate letters attached to numbers
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    
    # Process with spaCy
    doc = nlp(text)
    tokens = []
    for token in doc:
        # Skip punctuation if enabled
        if remove_punct and token.is_punct:
            continue
        # Skip stopwords if enabled
        if remove_stopwords and token.is_stop:
            continue
        # Keep alphabetic tokens or numbers
        if token.is_alpha or token.is_digit:
            tokens.append(token.lemma_.lower())
    
    return " ".join(tokens)