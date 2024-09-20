import nltk
from nltk.tokenize import word_tokenize
from typing import Dict, Any, List

class IELTSPreprocessor:
    name: str = "IELTS_Preprocessor"
    description: str = "Preprocesses an IELTS essay using NLTK for tokenization and POS tagging"

    def __init__(self):
        # Download necessary NLTK data
        resources = ['punkt', 'averaged_perceptron_tagger']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource)

        # Explicitly download punkt_tab
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')

    def run(self, essay: str) -> Dict[str, Any]:
        # NLTK processing
        tokens = self.safe_word_tokenize(essay)
        pos_tags = nltk.pos_tag(tokens)

        return {
            "tokens": tokens,
            "pos_tags": pos_tags
        }

    def safe_word_tokenize(self, text: str) -> List[str]:
        try:
            return word_tokenize(text)
        except LookupError:
            # If word_tokenize fails, use a simple split as fallback
            return text.split()

# Example usage
if __name__ == "__main__":
    preprocessor = IELTSPreprocessor()
    sample_essay = "This is a sample IELTS essay. It contains multiple sentences and various parts of speech."
    result = preprocessor.run(sample_essay)

    print("Tokens:", result["tokens"])
    print("POS Tags:", result["pos_tags"])