"""
Text Preprocessing Module
=========================
Handles all text cleaning and normalization steps:
- Lowercasing
- Special character / URL / email removal
- NLTK tokenization
- Stop-word removal
- Lemmatization
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception:
    print('Warning: NLTK resources could not be downloaded automatically. You may need to run NLTK downloads manually.')


class TextPreprocessor:
    """Full text preprocessing pipeline for NLP tasks."""

    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set(ENGLISH_STOP_WORDS)
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.lemmatizer.lemmatize('tests')
            self._lemmatizer_available = True
        except LookupError:
            self._lemmatizer_available = False
        # Compile regex patterns once for efficiency
        self._url_pattern = re.compile(r'http\S+|www\.\S+')
        self._email_pattern = re.compile(r'\S+@\S+')
        self._special_char_pattern = re.compile(r'[^a-zA-Z\s]')
        self._whitespace_pattern = re.compile(r'\s+')

    def _clean_text(self, text: str) -> str:
        """Apply all cleaning steps to a single text."""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = self._url_pattern.sub('', text)
        # Remove emails
        text = self._email_pattern.sub('', text)
        # Remove special characters and numbers
        text = self._special_char_pattern.sub(' ', text)
        # Normalize whitespace
        text = self._whitespace_pattern.sub(' ', text).strip()
        return text

    def _tokenize_and_process(self, text: str) -> str:
        """Tokenize, remove stop words, and lemmatize."""
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                if self._lemmatizer_available:
                    token = self.lemmatizer.lemmatize(token)
                processed_tokens.append(token)
        return ' '.join(processed_tokens)

    def preprocess(self, texts: list, show_progress: bool = True) -> list:
        """
        Preprocess a list of texts through the full pipeline.

        Parameters
        ----------
        texts : list of str
            Raw text documents.
        show_progress : bool
            Whether to show tqdm progress bar.

        Returns
        -------
        list of str
            Preprocessed text documents.
        """
        iterator = tqdm(texts, desc="Preprocessing texts", disable=not show_progress)
        processed = []
        for text in iterator:
            cleaned = self._clean_text(text)
            result = self._tokenize_and_process(cleaned)
            processed.append(result)
        return processed
