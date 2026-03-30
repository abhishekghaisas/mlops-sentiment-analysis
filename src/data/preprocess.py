"""Text preprocessing utilities for sentiment analysis."""

import re
import logging
from typing import List, Optional
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocess text data for sentiment analysis."""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        remove_special_chars: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_stopwords: Remove common stopwords
            lemmatize: Apply lemmatization
            remove_special_chars: Remove special characters
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_special_chars = remove_special_chars
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize tools
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
            
    def _download_nltk_data(self):
        """Download required NLTK datasets."""
        required_data = ['stopwords', 'wordnet', 'punkt', 'averaged_perceptron_tagger']
        
        for data in required_data:
            try:
                nltk.data.find(f'corpora/{data}')
            except LookupError:
                logger.info(f"Downloading NLTK data: {data}")
                nltk.download(data, quiet=True)
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        if self.remove_special_chars:
            text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return text.split()
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        if not self.remove_stopwords:
            return tokens
        
        return [word for word in tokens if word not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        if not self.lemmatize:
            return tokens
        
        return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Lemmatize
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Rejoin tokens
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str], show_progress: bool = True) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            show_progress: Show progress bar
            
        Returns:
            List of preprocessed texts
        """
        if show_progress:
            from tqdm import tqdm
            return [self.preprocess(text) for text in tqdm(texts, desc="Preprocessing")]
        else:
            return [self.preprocess(text) for text in texts]
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        output_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Preprocess text column in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            output_column: Name of output column (default: text_column + '_clean')
            
        Returns:
            DataFrame with preprocessed text
        """
        if output_column is None:
            output_column = f"{text_column}_clean"
        
        logger.info(f"Preprocessing {len(df)} texts...")
        
        df[output_column] = self.preprocess_batch(df[text_column].tolist())
        
        logger.info(f"Preprocessing complete. Results in column: {output_column}")
        
        return df


def validate_preprocessed_text(text: str) -> bool:
    """
    Validate that preprocessed text meets quality standards.
    
    Args:
        text: Preprocessed text
        
    Returns:
        True if valid, False otherwise
    """
    # Check if text is not empty
    if not text or len(text.strip()) == 0:
        return False
    
    # Check minimum length (at least 3 characters)
    if len(text) < 3:
        return False
    
    # Check that it contains at least one alphabetic character
    if not any(c.isalpha() for c in text):
        return False
    
    return True


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = TextPreprocessor()
    
    sample_texts = [
        "This movie was AMAZING! I loved every minute of it.",
        "Terrible film, waste of my time. Do NOT watch!",
        "It was okay, nothing special but not bad either.",
        "<html>Check out this site: http://example.com</html>"
    ]
    
    print("Original texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    
    print("\nPreprocessed texts:")
    for i, text in enumerate(sample_texts, 1):
        clean = preprocessor.preprocess(text)
        print(f"{i}. {clean}")
        print(f"   Valid: {validate_preprocessed_text(clean)}")