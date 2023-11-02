import os
import json
import string

import numpy as np

from data_accessor import DataAccessor
from nlp_preprocessor import NlpPreprocessor
from chat_dataset import ChatDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import NeuralNet


IGNORED_WORDS = string.punctuation

# Hyperparameters

BATCH_SIZE = 8

# Number of hidden layers used in our model
HIDDEN_SIZE = 8

LEARNING_RATE = 0.001
NUM_EPOCHS = 1000


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
        self._4_build_chat_dataset()
        self._5_define_model()
        self._6_loss_and_optimize()
        self._7_save_model_data()

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
        self.tags = sorted(set(self.tags))

    def _3_bag_of_words(self) -> None:
        for (pattern_sentence, tag) in self.xy:
            bag = self.nlp_preprocessor.bag_of_words(
                pattern_sentence, self.all_words)
            self.x_train.append(bag)

            label = self.tags.index(tag)
            self.y_train.append(label)

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)

    def _4_build_chat_dataset(self) -> None:
        self.dataset = ChatDataset(self.x_train, self.y_train)
        self.train_loader = DataLoader(
            dataset=self.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    def _5_define_model(self) -> None:
        print(
            f"Input size: {self.input_size()} = Number of known words: {len(self.all_words)}")
        print(f"Output size: {self.output_size()} = {self.tags}")
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuralNet(
            self.input_size(), HIDDEN_SIZE, self.output_size()).to(self.device)

    def _6_loss_and_optimize(self) -> None:
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS):
            for (words, labels) in self.train_loader:
                words = words.to(self.device)
                labels = labels.to(self.device)

                # forward
                outputs = self.model(words)
                loss = self.criterion(outputs, labels)

                # backward and optimize step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(
                    f"epoch {epoch + 1}/{NUM_EPOCHS}, loss={loss.item():.4f}")
        print(f"final loss, loss={loss.item():.4f}")

    def _7_save_model_data(self) -> None:
        data = {
            "model_state": self.model.state_dict(),
            "input_size": self.input_size(),
            "output_size": self.output_size(),
            "hidden_size": HIDDEN_SIZE,
            "all_words": self.all_words,
            "tags": self.tags
        }

        torch.save(data, os.path.join(
            DataAccessor.data_dirpath(), "model_data.pth"))
        print(f"Training complete, model data file saved")

    def forget_everything(self) -> None:
        self.all_words = []
        self.tags = []
        self.xy = []
        self.x_train = []
        self.y_train = []

    def input_size(self) -> int:
        """Total number of known words"""
        return len(self.x_train[0])

    def output_size(self) -> int:
        """Number of different classes/tags the model has to consider"""
        return len(self.tags)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_model()
