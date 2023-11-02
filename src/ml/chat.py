import random
import json
from typing import Any
import torch
from model import NeuralNet
from data_accessor import DataAccessor


class Chat:
    def __init__(self) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.intents = self._load_intents_file()
        self.model_data = self._load_model_data()
        self.input_size = self.model_data["input_size"]
        self.output_size = self.model_data["output_size"]
        self.hidden_size = self.model_data["hidden_size"]
        self.all_words = self.model_data["all_words"]
        self.tags = self.model_data["tags"]
        self.model_state = self.model_data["model_state"]
        self._load_model()

    def _load_intents_file(self) -> dict:
        with open(DataAccessor.intents_filepath(), "r") as f:
            intents = json.load(f)
        return intents

    def _load_model_data(self) -> Any:
        return torch.load(DataAccessor.model_data_filepath())

    def _load_model(self) -> None:
        self.model = NeuralNet(
            self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.model.load_state_dict(self.model_state)
        self.model.eval()


if __name__ == "__main__":
    chat = Chat()
