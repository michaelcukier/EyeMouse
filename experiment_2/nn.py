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
from PIL import Image
import numpy as np
from numpy import vstack
from numpy import sqrt
from sklearn.metrics import mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
from numpy import asarray
# https://stackoverflow.com/questions/32888108/denormalization-of-predicted-data-in-neural-networks

# def maximum_absolute_scaling(df):
#     df_scaled = df.copy()
#     df_scaled = df_scaled  / df_scaled.abs().max()
#     return df_scaled

myPD = pd.read_csv('./dlib_dataset2.csv')
myPD = asarray(myPD['x_coord']).reshape(-1, 1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(myPD)

#
# print(scaled, unscaled)
#
# quit()


def normalize(df):
    norm_df = df.copy()
    norm_df = (norm_df - norm_df.mean())/norm_df.std()
    m = norm_df.mean()
    s = norm_df.std()
    return (norm_df, m, s)

class MyDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        # self.y = self.annotations['x_coord']
        self.y = scaled
        # self.y = normalize(self.y)[0]
        # self.mean = normalize(self.y)[1]
        # self.std = normalize(self.y)[2]
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
        y_label = torch.tensor(int(self.y[index]))
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
    csv_file='dlib_dataset2.csv',
    root_dir='dlib_data'
)

train_set, test_set = torch.utils.data.random_split(dataset, lengths=[1600, 281])

train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=True)

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

model = ConvNet()
# criterion = nn.MSELoss()
criterion = RMSELoss

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

losses = []

for epoch in range(155):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        scores = model(data)
        loss = criterion(scores.float(), targets.float())
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0:
        print('Cost: {0} = {1}'.format(epoch, sum(losses)/len(losses)))


def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    mse = mean_squared_error(actuals, predictions)
    return mse

mse = evaluate_model(test_loader, model)
print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))

valll = asarray([sqrt(mse)]).reshape((-1, 1))
unscaled = scaler.inverse_transform(valll)
print('acc: ', unscaled)