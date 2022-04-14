from torch import nn
from torch.nn import Sequential, Conv1d, MaxPool1d, Flatten, Linear, ReLU, Softmax, Tanh


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model1 = Sequential(
            Conv1d(1, 16, kernel_size=3),
            ReLU(),
            # Conv1d(16, 16, kernel_size=3),
            MaxPool1d(3, padding=0),
            Conv1d(16, 64, kernel_size=4),
            ReLU(),
            # Conv1d(64, 64, kernel_size=3),
            MaxPool1d(3, padding=0),
            # Conv1d(64, 64, kernel_size=3),
            # Conv1d(64, 64, kernel_size=3),
            # MaxPool1d(3),
            Flatten(),
            Linear(27 * 64, 64),
            Linear(64, 3),
            # ReLU()
        )

    def forward(self, x):
        x = self.model1(x)
        return x
