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
torch.manual_seed(0)


def maximum_absolute_scaling(df):
    # copy the dataframe
    df_scaled = df.copy()
    # apply maximum absolute scaling
    df_scaled = df_scaled  / df_scaled.abs().max()
    return df_scaled

class MyDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        self.y = self.annotations['x_coord']
        self.y_labels = maximum_absolute_scaling(self.y)
        self.root_dir = root_dir


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        img = cv2.imread(img_path)
        res = cv2.resize(img, dsize=(40, 10), interpolation=cv2.INTER_CUBIC)
        myT = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        res = res.astype(np.float32)
        res = torch.from_numpy(res)
        res = myT(res)
        res = res.permute(2, 1, 0)
        y_label = torch.tensor(int(self.y_labels.iloc[index]))
        return (res, y_label)


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.fc2 = nn.Linear(360, 120)
        self.fc3 = nn.Linear(120, 1)

    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

dataset = MyDataset(
    csv_file='overfit_dataset.csv',
    root_dir='dlib_data'
)

train_set, test_set = torch.utils.data.random_split(dataset, lengths=[12, 1])

train_loader = DataLoader(dataset=train_set, batch_size=12, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=12, shuffle=True)

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

model = ConvNet()
criterion = nn.MSELoss()
# criterion = RMSELoss

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

losses = []

for epoch in range(100000):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        scores = model(data)
        loss = criterion(scores.float(), targets.float())
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 250 == 0:
        print('Cost: {0} = {1}'.format(epoch, sum(losses)/len(losses)))


from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set

        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse


mse = evaluate_model(train_loader, model)
print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))