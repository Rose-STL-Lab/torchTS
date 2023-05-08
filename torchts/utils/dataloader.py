#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import urllib.request
import zipfile
from pandas import HDFStore
import numpy as np
from os.path import join

# Define PyTorch dataset class
class BikeDataset(Dataset):
    # URL of the zip file to download
    url = "https://www.microsoft.com/en-us/research/uploads/prod/2016/02/Codes.zip"

    # Destination folder to extract the zip file
    destination_folder = "bike_sharing_dataset"
    
    def __init__(
        self,
        root:str,
        H:int,
        P:int,
        download: bool = False
    ):
        """
        H (int): amount of historical timestamp
        P (int): amount of predicted timestamp
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        
        X: historical data with shape 
        (number of samples, number of historical timestamp, entire traffic + check-out across 23 clusters + check-out proportion across 23 clusters + 7 features)
        Y: future data with shape
        (number of samples, number of futur timestamp,check-out across 23 clusters)
        """
        
        if download:
            # Download the zip file from the URL
            urllib.request.urlretrieve(self.url, "Codes.zip")

            # Extract the zip file to the destination folder
            with zipfile.ZipFile("Codes.zip", 'r') as zip_ref:
                zip_ref.extractall(root+"/"+self.destination_folder)

            print("Zip file downloaded and extracted successfully in "+root+"/"+self.destination_folder)

            # Load the .mat file
            mat = scipy.io.loadmat(root+"/"+self.destination_folder+'/Codes/Check-out Proportion/fea_pro.mat')
        else:
            # Load the .mat file
            mat = scipy.io.loadmat(root+'/Codes/Check-out Proportion/fea_pro.mat')

        # Get the 2D matrix
        fea = mat['fea']
        m_o = mat['m_o']
        m_O = mat['m_O']
        rho = mat['rho']
        
        # Split the data into input and output features
        fea_H = []
        m_o_H = []
        m_o_P = []
        m_O_H = []
        rho_H = []
        for i in range(H, len(fea) - P + 1):
            fea_H.append(fea[i-H:i])
            m_o_H.append(m_o[i-H:i])
            m_o_P.append(m_o[i:i+P])
            m_O_H.append(m_O[0][i-H:i])
            rho_H.append(rho[i-H:i])
        fea_H = torch.tensor(fea_H, dtype=torch.float32)
        m_o_H = torch.tensor(m_o_H, dtype=torch.float32)
        m_O_H = torch.tensor(m_O_H, dtype=torch.float32)
        m_O_H = m_O_H.unsqueeze(-1)
        rho_H = torch.tensor(rho_H, dtype=torch.float32)
        
        # Concetenate four tensors
        X = torch.cat([m_O_H, m_o_H, rho_H, fea_H], dim=2)
        
        #print(X.shape)
        self.x = X
        self.y = torch.tensor(m_o_P, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class seq2seq:
    # URL of the zip file to download
    # url = "https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
    
    def __init__(
        self,
        root:str,
        horizon=12, 
        window=1, 
        features=['avg_speed'],#['avg_speed', 'total_flow']
        test_ratio = 0.2,
        train_ratio = 0.7,
    ):
        """
        """
        filename = root+"/pems-bay.h5"

        # read store for HDFS
        store = HDFStore(filename)
        store
        
        # select a fraction from all stations
        denom = 30 # (i.e. 1/30)
        df_query_raw = store.speed.stack(
                                ).reset_index(name='avg_speed'
                                ).rename(columns={"level_0": "timestamp",
                                                  "sensor_id": "station"})
        df_query_raw = store.speed.loc[:, df_query_raw.station.unique()[:len(df_query_raw.station.unique())//denom]].stack(
                                                    ).reset_index(name='avg_speed'
                                                    ).rename(columns={"level_0": "timestamp",
                                                                      "sensor_id": "station"})
        print(df_query_raw.shape)
        print('Number of stations {}'.format(len(df_query_raw.station.unique())))
        df_query_raw.head()
        print("1")
        #clean Data
        df_query_raw.isna().sum() # check for na values
        # fill na values with rolling mean
        df_query_cleaned = df_query_raw.fillna(df_query_raw.rolling(window=6,min_periods=1).mean())
        df_query_cleaned.isna().sum() # verify na values are not present
        print("2")
        #Generate Seq2Seq Sequences
        idx_cols = ['station','timestamp']

        df_query_cleaned = df_query_cleaned.set_index(idx_cols).sort_values(by=idx_cols)
        print("3")
        features_tensor_list = []
        for f in features:
            print(f)

            ts_seq_list = []
            for s in df_query_cleaned.index.unique(level=0): # concatenate stations next to each other
    #             print(s)
                values = df_query_cleaned.loc[s][f].values

                for i in range(len(values)-horizon*2):
                    arr = np.array(values[i:i+horizon*2])
                    ts_seq_list.append(torch.from_numpy(arr.reshape(horizon*2,1)))

            sequence_tensor = torch.stack(ts_seq_list, dim=0)

            features_tensor_list.append(sequence_tensor)
        print("4")
        data_seq2seq = torch.cat(features_tensor_list, dim=2)
        print("fin")
        print(data_seq2seq.shape)
        
        # generate x and y vectors
        x = data_seq2seq[:, :12, :]
        y = data_seq2seq[:, 12:, :]

        print(x.shape, y.shape)
        
        # define split ratio for train, val and test
        num_samples = x.shape[0]
        num_test = round(num_samples * test_ratio)#test ratio
        num_train = round(num_samples * train_ratio)#train ratio
        num_val = num_samples - num_test - num_train

        print('Total number of samples: {}'.format(num_samples))
        print('Percentage for train: {:.2f}'.format(100*num_train/num_samples))
        print('Percentage for val: {:.2f}'.format(100*num_val/num_samples))
        print('Percentage for test: {:.2f}'.format(100*num_test/num_samples))
        
        # train
        x_train = x[:num_train]
        y_train = y[:num_train]

        # val
        x_val = x[num_train: num_train + num_val] 
        y_val = y[num_train: num_train + num_val]

        # test
        x_test = x[-num_test:] 
        y_test = y[-num_test:]

        print(x_train.shape, y_train.shape)
        print(x_val.shape, y_val.shape)
        print(x_test.shape, y_test.shape)
        
        # output train, val, and test NPZ files
        output_dir = root

        for cat in ["train", "val", "test"]:

            _x = locals()["x_" + cat] 
            _y = locals()["y_" + cat]

            print(cat, "x: ", _x.shape, "y:", _y.shape)

            np.savez_compressed(
                join(output_dir, "%s.npz" % cat),
                x=_x,
                y=_y
            )


# # Create PyTorch dataset and data loader
# dataset = BikeDataset(root = "C:/Users/Felix/Documents/Spatiotemporal_Data_Repository/bike_test",H = 3,P = 2,download = True)
# 
# print(dataset.x.size())
# print(dataset.y.size())
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# 
# # define neural network model
# class BikeModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = torch.nn.Linear(3*(7+23+1+23), 128)
#         self.fc2 = torch.nn.Linear(128, 64)
#         self.fc3 = torch.nn.Linear(64, 2*23)
#     
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)  # flatten the input tensor
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x.view(x.shape[0], 2, 23)  # reshape the output tensor
# 
# # define loss function and optimizer
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(BikeModel().parameters())
# 
# # train the model
# num_epochs = 10
# for epoch in range(num_epochs):
#     epoch_loss = 100.0
#     for batch_idx, (data, target) in enumerate(dataloader):
#         optimizer.zero_grad()
#         output = BikeModel()(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     print(f"Epoch {epoch+1} loss: {epoch_loss/len(dataloader):.4f}")
# 

# dataset = seq2seq(root = ".",horizon=12, window=1, features=['avg_speed'],test_ratio = 0.2,train_ratio = 0.7)
