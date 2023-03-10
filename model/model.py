import torch.nn as nn
import torch.nn.functional as F

# ニューラルネットワークの定義
class CNN(nn.Module):
    def __init__(self, grayed):
        super(CNN, self).__init__()
        if grayed:
            self.conv1 = nn.Conv2d(1, 6, 5)
        else:
            self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 157 * 157, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 16 * 157 * 157)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x