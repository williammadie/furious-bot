import nltk
from nltk.stem.snowball import SnowballStemmer

import numpy as np


class NlpPreprocessor:
    def __init__(self) -> None:
        self.stemmer = SnowballStemmer(language="french")

    def tokenize(self, sentence: str) -> list:
        return nltk.word_tokenize(sentence, language="french")

    def stem(self, word: str) -> str:
        return self.stemmer.stem(word.lower())

    def bag_of_words(self, tokenized_sentence: list, all_words: list) -> list:
        """Turn a tokenized_sentence into a bag_of_words 
        
        See example below:
        tokenized_sentence = ["hello", "how", "are", "you"]
        all_words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bag_of_words = [0,    1   ,  0 ,   1  ,   0  ,    0   ,    0  ]
        """
        tokenized_sentence = [self.stem(w) for w in tokenized_sentence]
        bag = np.zeros(len(all_words), dtype=np.float32)
        for idx, w in enumerate(all_words):
            if w in tokenized_sentence:
                bag[idx] = 1.0
        
        return bag


if __name__ == "__main__":
    a = "Hello guys, do you want to drink a beer tonight?"
    nlp_preprocessor = NlpPreprocessor()
    a = nlp_preprocessor.tokenize(a)
    print(a)
