import numpy as np
import pandas as pd
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
import yaml
from .normalize import normalize_pose_landmarks

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


# We will use dynamic window_size
WINDOW_SIZE = 30 # ex: 30 frames per block  
# We have 6 output action classes.
TOT_ACTION_CLASSES = 8

class PoseDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PoseDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.train_data_path = self.data_root + "train_data.csv"
        self.train_info_path = self.data_root + "train_info.csv"
        self.test_data_path = self.data_root + "test_data.csv"
        self.test_info_path = self.data_root + "test_info.csv"
        self.load(self.train_data_path, self.train_info_path)

    def preprocess_data(self, x, block_sizes):

        # 1. delete the score columns
        # x = np.delete(x, slice(2, None, 3), 1)
        # 2. nomalize the data
        # x = normalize_pose_landmarks(x)
        # 3. convert all NAN to -10
        x = np.nan_to_num(x, copy=False, nan=-10, posinf=None, neginf=None)
        # 4. convert to numpy float array
        x = x.astype(np.float32)
        # 5. fill the blocks with padding
        # we will process using collete_fn
        ''' handmade solution
        final_x = np.array([]).reshape(0, x.shape[1])
        i_iter = 0
        for block_size in block_sizes:
            block = np.full((self.window_size, x.shape[1]), -10)
            for i in range(block_size):
                block[i] = x[i_iter + i]
            i_iter += block_size
            final_x = np.concatenate((final_x, block), axis=0)
        # 5. split the data into blocks
        blocks = int(len(final_x) / self.window_size)
        np.split(final_x, blocks)
        '''
        # 6. split the data into blocks
        x = np.split(x, np.cumsum(block_sizes))
        
        return x
    
    def load(self, data_path, info_path):
        
        global WINDOW_SIZE
        data = pd.read_csv(data_path, sep=',')
        info = pd.read_csv(info_path, sep=',', header=None)
        y =  []
        # calculate the number of action classes and find the largest block
        block_sizes = []
        action_classes_num = 0
        data_dict = {}
        for row in info.iterrows():
            if pd.isna(row[1][3]):
                data_dict[row[1][1]-1] = row[1][0]
                y += [row[1][1]] * row[1][2]
                action_classes_num += 1
                continue
            block_sizes.append(int(row[1][3]))
        
        # Convert the dictionary to YAML format
        yaml_output = yaml.dump(data_dict, default_flow_style=False)

        # Write the YAML output to a file
        with open('src/labels.yaml', 'w') as yamlfile:
            yamlfile.write(yaml_output)

        if TOT_ACTION_CLASSES != action_classes_num:
            raise ValueError("The number of action classes is not equal to the number of classes in the data")
        WINDOW_SIZE = max(block_sizes) if WINDOW_SIZE == 0 else WINDOW_SIZE
        # TODO: test if normalize work 2024/4/12: seems work
        x = self.preprocess_data(data.values, block_sizes)
        return x, np.array(y) - 1
    
    def collate_fn(self, batch):
        # batch 是一個列表，其中包含了**所有** `__getitem__` 的輸出
        # 我們需要對這些輸出應用 pad_sequence

        (x, y) = zip(*batch)
        x = [torch.tensor(i, dtype=torch.float) for i in x]
        y = torch.tensor(y, dtype=torch.long)
        x_lens = [len(x) for x in x]
        x_pad = pad_sequence(x, batch_first=True, padding_value=-10)

        return x_pad, y, x_lens
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Please load the train first and then the test data
        X_train,y_train = self.load(self.train_data_path, self.train_info_path)
        X_test, y_test = self.load(self.test_data_path, self.test_info_path)
        self.train_dataset = PoseDataset(X_train, y_train)
        self.val_dataset = PoseDataset(X_test, y_test)

    def train_dataloader(self):
        # train loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        return train_loader

    def val_dataloader(self):
        # validation loader
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        return val_loader
    
    def __getattribute__(self, name: np.str):
        return super().__getattribute__(name)


#lstm classifier definition
class ActionClassificationLSTM(pl.LightningModule):
    # initialise method
    def __init__(self, input_features, hidden_dim, learning_rate=0.001):
        super().__init__()
        # save hyperparameters
        self.save_hyperparameters()
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # TODO: add mask to ignore padding
        self.lstm = nn.LSTM(input_features, hidden_dim, batch_first=True)
        # The linear layer that maps from hidden state space to classes
        self.linear = nn.Linear(hidden_dim, TOT_ACTION_CLASSES)

    def forward(self, x):
        # invoke lstm layer
        lstm_out, (ht, ct) = self.lstm(x)
        # invoke linear layer
        return self.linear(ht[-1])

    def training_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y, x_lens = batch
        # x = torch.unsqueeze(x, dim=2)           
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        # reduce dimension
        # y = torch.squeeze(y)
        # convert to long
        # y = y.long()
        # get prediction
        y_pred = self(x)
        # calculate loss
        loss = F.cross_entropy(y_pred, y)
        # get probability score using softmax
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        # calculate accuracy
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {
            'batch_train_loss': loss,
            'batch_train_acc': acc
        }
        # log the metrics for pytorch lightning progress bar or any other operations
        self.log('batch_train_loss', loss, prog_bar=True)
        self.log('batch_train_acc', acc, prog_bar=True)
        #return loss and dict
        return {'loss': loss, 'result': dic}

    def training_epoch_end(self, training_step_outputs):
        # calculate average training loss end of the epoch
        avg_train_loss = torch.tensor([x['result']['batch_train_loss'] for x in training_step_outputs]).mean()
        # calculate average training accuracy end of the epoch
        avg_train_acc = torch.tensor([x['result']['batch_train_acc'] for x in training_step_outputs]).mean()
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('train_loss', avg_train_loss, prog_bar=True)
        self.log('train_acc', avg_train_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y, x_lens = batch
        # x = torch.unsqueeze(x, dim=2)           
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        # reduce dimension
        # y = torch.squeeze(y)
        # convert to long
        # y = y.long()
        # get prediction
        y_pred = self(x)
        # calculate loss
        loss = F.cross_entropy(y_pred, y)
        # get probability score using softmax
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        # calculate accuracy
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {
            'batch_val_loss': loss,
            'batch_val_acc': acc
        }
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('batch_val_loss', loss, prog_bar=True)
        self.log('batch_val_acc', acc, prog_bar=True)
        #return dict
        return dic

    def validation_epoch_end(self, validation_step_outputs):
        # calculate average validation loss end of the epoch
        avg_val_loss = torch.tensor([x['batch_val_loss']
                                     for x in validation_step_outputs]).mean()
        # calculate average validation accuracy end of the epoch
        avg_val_acc = torch.tensor([x['batch_val_acc']
                                    for x in validation_step_outputs]).mean()
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('val_loss', avg_val_loss, prog_bar=True)
        self.log('val_acc', avg_val_acc, prog_bar=True)

    def configure_optimizers(self):
        # adam optimiser
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # learning rate reducer scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-15, verbose=True)
        # scheduler reduces learning rate based on the value of val_loss metric
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val_loss"}}
