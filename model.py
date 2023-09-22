import torch.nn as nn
import torch.nn.functional as F
import torch
class SimpleNN4(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleNN4, self).__init__()
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(21 * 3 * 2, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(32, 16)
        self.dropout5 = nn.Dropout(0.1)
        self.fc6 = nn.Linear(16, num_classes)
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
    
if __name__ == "__main__":
    a = torch.rand(1, 1, 126)
    model = SimpleNN4()
    print(model(a).shape)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")