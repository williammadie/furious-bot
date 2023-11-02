"""This module describes the neural network used for the chatbot

We use the Feed Forward Neural Network model for the chat bot.
"""

import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super(NeuralNet, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.layer_3(out)
        out = self.relu(out)
        # No activation and no softmax
        return out
