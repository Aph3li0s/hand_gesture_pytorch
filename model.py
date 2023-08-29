import torch.nn as nn
class SimpleNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleNN, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(21 * 3, 20)
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
