import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

def custom_tokenize(text):
    sentences = sent_tokenize(text)
    return [word_tokenize(sentence) for sentence in sentences]

import language_tool_python
import textstat
from wordfreq import word_frequency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from typing import Dict, Any

class IELTSEssayAnalyzer:
    name: str = "IELTS_Essay_Analyzer"
    description: str = "Analyzes an IELTS essay for grammar, readability, vocabulary, and coherence"

    def __init__(self):
        self._language_tool = language_tool_python.LanguageTool('en-US')

    def run(self, essay: str) -> Dict[str, Any]:
        # Grammar checking
        grammar_errors = self._language_tool.check(essay)

        # Readability analysis
        readability_score = textstat.flesch_reading_ease(essay)

        # Vocabulary assessment
        words = [word.lower() for sentence in custom_tokenize(essay) for word in sentence if word not in string.punctuation]
        sentences = sent_tokenize(essay)

        # Coherence analysis (sentence similarity)
        sentences = nltk.sent_tokenize(essay)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        coherence_score = sum(similarity_matrix.flatten()) / (len(sentences) ** 2)

        return {
            "grammar_errors": [str(error) for error in grammar_errors],
            "readability_score": readability_score,
            "vocabulary_score": 1 - vocabulary_score,  # Invert the score so higher is better
            "coherence_score": coherence_score
        }