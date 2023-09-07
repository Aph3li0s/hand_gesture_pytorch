import torch.nn as nn
import torch.nn.functional as F
class SimpleNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleNN, self).__init__()
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(21 * 3 * 2, 80)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(80, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 20)
        self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(20, 10)
        self.dropout5 = nn.Dropout(0.1)
        self.fc6 = nn.Linear(10, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)
        x = self.fc6(x)
        return x


class SimpleNN2(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleNN2, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(21 * 3, 40)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(40, 20)
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x

class SimpleNN3(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleNN3, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(21 * 3 * 2, 63)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(63, 40)
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x

class SimpleNN4(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleNN4, self).__init__()
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(21 * 3 * 2, 80)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(80, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 20)
        self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(20, 10)
        self.dropout5 = nn.Dropout(0.1)
        self.fc6 = nn.Linear(10, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)
        x = self.fc6(x)
        return x
