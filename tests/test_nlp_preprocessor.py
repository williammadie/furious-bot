import numpy as np
import unittest


from src.ml.nlp_preprocessor import NlpPreprocessor


class TestCalculations(unittest.TestCase):

    def test_bag_of_words(self):
        tokenized_sentence = ["hello", "how", "are", "you"]
        all_words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bag_of_words = [0, 1, 0, 1, 0, 0, 0]
        nlp_prep = NlpPreprocessor()
        res_bag = nlp_prep.bag_of_words(tokenized_sentence, all_words)
        self.assertTrue(np.array_equal(res_bag, bag_of_words))


if __name__ == '__main__':
    unittest.main()
