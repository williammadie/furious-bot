from torch.utils.data import Dataset


class ChatDataset(Dataset):
    def __init__(self, x_train, y_train) -> None:
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
        super().__init__()

    def __getitem__(self, idx) -> tuple:
        return self.x_data[idx], self.y_data[idx]

    def __len__(self) -> int:
        return self.n_samples
