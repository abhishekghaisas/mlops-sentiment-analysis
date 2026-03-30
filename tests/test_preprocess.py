"""Tests for text preprocessing."""

import pytest
import pandas as pd
from src.data.preprocess import TextPreprocessor, validate_preprocessed_text


class TestTextPreprocessor:
    """Test suite for TextPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance for testing."""
        return TextPreprocessor(
            lowercase=True,
            remove_stopwords=True,
            lemmatize=True,
            remove_special_chars=True
        )
    
    def test_clean_text_lowercase(self, preprocessor):
        """Test that text is converted to lowercase."""
        text = "This Movie Is AMAZING!"
        result = preprocessor.clean_text(text)
        assert result.islower()
    
    def test_clean_text_html_removal(self, preprocessor):
        """Test HTML tag removal."""
        text = "<p>Great movie!</p>"
        result = preprocessor.clean_text(text)
        assert "<p>" not in result
        assert "</p>" not in result
        assert "great movie" in result
    
    def test_clean_text_url_removal(self, preprocessor):
        """Test URL removal."""
        text = "Check this out http://example.com great!"
        result = preprocessor.clean_text(text)
        assert "http://example.com" not in result
        assert "check" in result
        assert "great" in result
    
    def test_clean_text_email_removal(self, preprocessor):
        """Test email address removal."""
        text = "Contact me at test@example.com for details"
        result = preprocessor.clean_text(text)
        assert "test@example.com" not in result
    
    def test_clean_text_special_chars_removal(self, preprocessor):
        """Test special character removal."""
        text = "Amazing!!! @#$%"
        result = preprocessor.clean_text(text)
        assert "!" not in result
        assert "@" not in result
        assert "#" not in result
        assert "amazing" in result
    
    def test_clean_text_extra_whitespace(self, preprocessor):
        """Test extra whitespace removal."""
        text = "This    is   a    test"
        result = preprocessor.clean_text(text)
        assert "  " not in result
        assert result.count(" ") == 3  # Single spaces only
    
    def test_tokenize(self, preprocessor):
        """Test tokenization."""
        text = "this is a test"
        tokens = preprocessor.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) == 4
        assert tokens == ["this", "is", "a", "test"]
    
    def test_remove_stopwords(self, preprocessor):
        """Test stopword removal."""
        tokens = ["this", "is", "a", "great", "movie"]
        result = preprocessor.remove_stopwords_from_tokens(tokens)
        # 'this', 'is', 'a' are stopwords
        assert "great" in result
        assert "movie" in result
        assert "this" not in result
        assert "is" not in result
        assert "a" not in result
    
    def test_lemmatize_tokens(self, preprocessor):
        """Test lemmatization."""
        tokens = ["running", "ran", "runs"]
        result = preprocessor.lemmatize_tokens(tokens)
        # All should be lemmatized to 'run'
        assert all(token in ["run", "running", "ran"] for token in result)
    
    def test_preprocess_full_pipeline(self, preprocessor):
        """Test full preprocessing pipeline."""
        text = "This Movie Was AMAZING!!! I loved it. http://example.com"
        result = preprocessor.preprocess(text)
        
        # Should be lowercase
        assert result.islower()
        
        # Should not contain special characters
        assert "!" not in result
        
        # Should not contain URL
        assert "http" not in result
        
        # Should contain meaningful words
        assert "movie" in result or "amazing" in result or "loved" in result
    
    def test_preprocess_empty_string(self, preprocessor):
        """Test preprocessing empty string."""
        result = preprocessor.preprocess("")
        assert result == ""
    
    def test_preprocess_none_input(self, preprocessor):
        """Test preprocessing None input."""
        result = preprocessor.preprocess(None)
        assert result == ""
    
    def test_preprocess_batch(self, preprocessor):
        """Test batch preprocessing."""
        texts = [
            "Great movie!",
            "Terrible film.",
            "It was okay."
        ]
        results = preprocessor.preprocess_batch(texts, show_progress=False)
        
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
        assert all(r.islower() for r in results if r)
    
    def test_preprocess_dataframe(self, preprocessor):
        """Test DataFrame preprocessing."""
        df = pd.DataFrame({
            'text': ['Great movie!', 'Bad film!', 'Okay.'],
            'label': [1, 0, 0]
        })
        
        result_df = preprocessor.preprocess_dataframe(df, text_column='text')
        
        assert 'text_clean' in result_df.columns
        assert len(result_df) == 3
        assert all(result_df['text_clean'].str.islower())
    
    def test_preprocess_preserves_meaning(self, preprocessor):
        """Test that preprocessing preserves semantic meaning."""
        positive = "This is an absolutely fantastic and amazing movie!"
        negative = "This is a terrible and horrible movie!"
        
        pos_result = preprocessor.preprocess(positive)
        neg_result = preprocessor.preprocess(negative)
        
        # Positive sentiment words should remain
        assert any(word in pos_result for word in ['fantastic', 'amazing', 'great'])
        
        # Negative sentiment words should remain
        assert any(word in neg_result for word in ['terrible', 'horrible', 'bad'])


class TestValidatePreprocessedText:
    """Test suite for text validation."""
    
    def test_validate_valid_text(self):
        """Test validation of valid text."""
        assert validate_preprocessed_text("this is valid text")
    
    def test_validate_empty_string(self):
        """Test validation of empty string."""
        assert not validate_preprocessed_text("")
    
    def test_validate_whitespace_only(self):
        """Test validation of whitespace-only string."""
        assert not validate_preprocessed_text("   ")
    
    def test_validate_too_short(self):
        """Test validation of very short text."""
        assert not validate_preprocessed_text("ab")
    
    def test_validate_no_letters(self):
        """Test validation of text with no letters."""
        assert not validate_preprocessed_text("123 456")
    
    def test_validate_minimum_length(self):
        """Test validation of minimum length text."""
        assert validate_preprocessed_text("abc")


class TestPreprocessorConfiguration:
    """Test different preprocessor configurations."""
    
    def test_no_lowercase(self):
        """Test preprocessor without lowercasing."""
        preprocessor = TextPreprocessor(lowercase=False)
        text = "Great Movie"
        result = preprocessor.clean_text(text)
        assert not result.islower()
        assert "Great" in result or "Movie" in result
    
    def test_keep_stopwords(self):
        """Test preprocessor that keeps stopwords."""
        preprocessor = TextPreprocessor(remove_stopwords=False)
        text = "this is a great movie"
        result = preprocessor.preprocess(text)
        # Stopwords should be present
        assert any(word in result.split() for word in ["this", "is", "a"])
    
    def test_no_lemmatization(self):
        """Test preprocessor without lemmatization."""
        preprocessor = TextPreprocessor(lemmatize=False)
        text = "running movies"
        result = preprocessor.preprocess(text)
        # Words should not be lemmatized
        assert "running" in result or "movie" in result


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])