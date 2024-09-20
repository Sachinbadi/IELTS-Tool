from Preprocessing_tool import IELTSPreprocessor
from Analysis_node1 import IELTSEssayAnalyzer

def test_ielts_tools():
    # Sample IELTS essay
    sample_essay = """
    In recent years, the popularity of social media has grown exponentially, transforming the way people communicate and share information. While some argue that social media has enhanced global connectivity and access to diverse perspectives, others contend that it has led to decreased face-to-face interactions and increased social isolation. This essay will examine both viewpoints and provide a balanced analysis of the impact of social media on modern society.
    """

    print("Starting tool tests...")

    # Test Preprocessor
    try:
        print("\nTesting IELTS Preprocessor:")
        preprocessor = IELTSPreprocessor()
        preprocessor_result = preprocessor.run(sample_essay)

        if "tokens" in preprocessor_result and "pos_tags" in preprocessor_result:
            print("✓ Preprocessor successfully tokenized the essay.")
            print(f"  First 5 tokens: {preprocessor_result['tokens'][:5]}")
            print(f"  First 5 POS tags: {preprocessor_result['pos_tags'][:5]}")
        else:
            print("✗ Preprocessor failed to produce expected output.")
    except Exception as e:
        print(f"✗ Error in Preprocessor: {str(e)}")

    print("\n" + "="*50 + "\n")

    # Test Analyzer
    try:
        print("Testing IELTS Essay Analyzer:")
        analyzer = IELTSEssayAnalyzer()
        analyzer_result = analyzer.run(sample_essay)

        expected_keys = ["grammar_errors", "readability_score", "vocabulary_score", "coherence_score"]
        if all(key in analyzer_result for key in expected_keys):
            print("✓ Analyzer successfully processed the essay.")
            for key in expected_keys:
                print(f"  {key.replace('_', ' ').title()}: {analyzer_result[key]}")
        else:
            print("✗ Analyzer failed to produce expected output.")
            print(f"  Expected keys: {expected_keys}")
            print(f"  Actual keys: {list(analyzer_result.keys())}")
    except Exception as e:
        print(f"✗ Error in Analyzer: {str(e)}")

if __name__ == "__main__":
    test_ielts_tools()