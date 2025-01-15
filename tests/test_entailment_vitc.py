import unittest
import pandas as pd
import nltk
from src.entailment_vitc import load_model_and_tokenizer, NLI, tokenize_sentences, find_span_indices, load_vp_data

class TestEntailmentVitc(unittest.TestCase):

    def setUp(self):
        # Setup any state specific to the test suite
        self.model_name = "tals/albert-xlarge-vitaminc"
        self.data_dir = "./data"  # Adjust this path to where your data is located
        self.fname = "questions_and_human_perspectives_with_responses.csv"

    def test_load_model_and_tokenizer(self):
        # Test if the model and tokenizer load successfully
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_name)
        self.assertIsNotNone(self.tokenizer, "Tokenizer should not be None")
        self.assertIsNotNone(self.model, "Model should not be None")

    def test_nli_function(self):
        # Test the NLI function with sample inputs
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_name)
        text_a = "The sky is blue."
        text_b = "The sky is clear."
        label, score = NLI(text_a, text_b, self.model, self.tokenizer)
        self.assertIn(label, ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"], "Label should be a valid NLI label")
        self.assertIsInstance(score, float, "Score should be a float")

    def test_tokenize_sentences(self):
        # Test sentence tokenization
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        text = "This is a sentence. This is another one."
        sentences = tokenize_sentences(text)
        self.assertEqual(len(sentences), 2, "There should be two sentences")

    def test_find_span_indices(self):
        # Test finding span indices
        string = "This is a test string."
        substring = "test"
        span = find_span_indices(string, substring)
        self.assertEqual(span, (10, 14), "Span indices should be correct")

    def test_load_vp_data_full(self):
        # Test loading the full dataset
        df = load_vp_data(self.data_dir, fname=self.fname)
        self.assertIsInstance(df, pd.DataFrame, "The result should be a DataFrame")
        self.assertGreater(len(df), 0, "The DataFrame should not be empty")
        self.assertIn('source', df.columns, "DataFrame should have a 'source' column")

    def test_load_vp_data_sample(self):
        # Test loading a sample of the dataset
        sample_size = 10
        df = load_vp_data(self.data_dir, n=sample_size, seed=42, fname=self.fname)
        self.assertIsInstance(df, pd.DataFrame, "The result should be a DataFrame")
        self.assertEqual(len(df), sample_size, f"The DataFrame should have {sample_size} rows")
        self.assertIn('source', df.columns, "DataFrame should have a 'source' column")


if __name__ == '__main__':
    unittest.main()