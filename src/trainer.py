import os
import json
import string

import numpy as np

import torch
import torch.nn as nn

from data_accessor import DataAccessor
from nlp_preprocessor import NlpPreprocessor

IGNORED_WORDS = string.punctuation


class Trainer:
    def __init__(self) -> None:
        self.intents = self._load_intents_file()
        self.all_words = []
        self.tags = []
        self.xy = []
        self.x_train = []
        self.y_train = []
        self.nlp_preprocessor = NlpPreprocessor()

    def _load_intents_file(self) -> dict:
        with open(DataAccessor.intents_filepath(), "r") as f:
            intents = json.load(f)
        return intents

    def train_model(self) -> None:
        self._1_create_training_data()
        self._2_lower_and_stem()
        self._3_bag_of_words()

    def _1_create_training_data(self) -> None:
        for intent in self.intents["intents"]:
            tag = intent["tag"]
            self.tags.append(tag)
            for pattern in intent["patterns"]:
                w = self.nlp_preprocessor.tokenize(pattern)
                self.all_words.extend(w)
                self.xy.append((w, tag))

    def _2_lower_and_stem(self) -> None:
        self.all_words = [self.nlp_preprocessor.stem(
            w) for w in self.all_words if w not in IGNORED_WORDS]
        self.all_words = sorted(set(self.all_words))
        tags = sorted(set(tags))
    
    def _3_bag_of_words(self) -> None:
        for (pattern_sentence, tag) in self.xy:
            bag = self.nlp_preprocessor.bag_of_words(pattern_sentence, self.all_words)
            self.x_train.append(bag)

            label = self.tags.index(tag)
            self.y_train.append(label)
        
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)

    def forget_everything(self) -> None:
        self.all_words = []
        self.tags = []
        self.xy = []
        self.x_train = []
        self.y_train = []


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_model()
    print(trainer.all_words)
