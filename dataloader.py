import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from PIL import Image
import numpy as np


class MyDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.annotations)  # +- 500

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])

        img = cv2.imread(img_path)
        res = cv2.resize(img, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
        res = res.astype(np.float32)  # you should add this line
        res = torch.from_numpy(res)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        return (res, y_label)



class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(1875, 200)
        self.fc2 = nn.Linear(200, 20)
        self.fc3 = nn.Linear(20, 1)


    def forward(self, x):
        x = x.float()
        x = x.permute(0, 3, 1, 2)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



dataset = MyDataset(
    csv_file='dataset.csv',
    root_dir='tmp'
)

train_set, test_set = torch.utils.data.random_split(dataset, lengths=[500, 70])

train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

model = ConvNet()
criterion = RMSELoss

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(50):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):

        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores.float(), targets.float())
        losses.append(loss.item())

        # backward
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


    print('Cost: {0} = {1}'.format(epoch, sum(losses)/len(losses)))




# tasks:

# - add more data
# - build net working with two images
# - add the head x/y coordinates
# - how to search for optimal architecture
# - normalizing the images
# -


