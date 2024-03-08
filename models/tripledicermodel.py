"""
The scripts to carry on all the machine learning tasks from dataset building up to training and predicting.
"""
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, optim
from tqdm import tqdm
from utils import readit
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics.regression import MeanSquaredError
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

"""
Building up three deep learning models.
"""
# Simple Fully connected Neutral Network build using lightning
class Simple_Fully_Connected_Neutral_Network(L.LightningModule):
    """ Simply Fully Connected Neutral Network"""
    def __init__(self, encoder, decoder, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder
        self.learning_rate = learning_rate

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# CNN network
class CNN_Dicer(pl.LightningModule):
    """CNN network model for dicing"""

    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        # Input shape (32, 1, 8, 8)
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # Output shape (32, 1, 9, 9)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        # Output shape (32, 1, 8, 8)?
        self.conv_layer2 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # Output shape (32, 4, 9, 9)
        self.fully_connected_1 = nn.Linear(in_features=4*9*9, out_features=300)
        self.fully_connected_2 = nn.Linear(in_features=300, out_features=150)
        self.fully_connected_3 = nn.Linear(in_features=150, out_features=70)
        self.fully_connected_4 = nn.Linear(in_features=70, out_features=2)

        self.mean_square_error = MeanSquaredError(num_outputs=2)
        self.loss = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, input):
        output = self.conv_layer1(input)
        output = self.relu1(output)
        output = self.pool(output)

        output = self.conv_layer2(output)
        output = self.relu2(output)

        output = output.view(-1, 4*9*9)
        output = self.fully_connected_1(output)
        output = self.fully_connected_2(output)
        output = self.fully_connected_3(output)
        output = self.fully_connected_4(output)

        return output

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.view(inputs.shape[0], 1, 8, 8)  # Turn the inputs into image like data
        outputs = self(inputs)
        train_accuracy = self.mean_square_error(outputs, targets)
        loss = self.loss(outputs, targets)
        self.log('train_accuracy', train_accuracy, prog_bar=True)
        self.log('train_loss', loss)
        return {"loss":loss, "train_accuracy":train_accuracy}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.view(inputs.shape[0], 1, 8, 8) # Turn the inputs into image like data
        outputs = self.forward(inputs)
        test_accuracy = self.mean_square_error(outputs, targets)
        loss = self.loss(outputs, targets)
        return {"test_loss":loss, "test_accuracy":test_accuracy}

    def predict_step(self, batch, batch_idx):
        return self(batch)

# LSTM neutral network
class LSTM_Neutral_Network(L.LightningModule):
    def __init__(self, input_size=64, output_size=2, hidden_dim=32, n_layers=2, learning_rate=1e-3):
        super(LSTM_Neutral_Network, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, num_layers=self.n_layers,
                            bidirectional=False, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

        self.loss = nn.MSELoss()

        self.learning_rate = learning_rate

    def get_hidden(self, batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        cell_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        hidden = (hidden_state, cell_state)
        return hidden

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.get_hidden(batch_size)
        out, hidden = self.lstm(x, hidden)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        features, targets = train_batch
        outputs = self(features)
        outputs = outputs.view(-1)
        loss = self.loss(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        features, targets = val_batch
        outputs = self(features)
        outputs = outputs.view(-1)
        loss = self.loss(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)


"""
Preparing the dataset.
"""
import utils.readrawdata as rrd

"""
Do the modeling and the prediction
"""

class Triple_Dicer:
    """Implement the triple dicer model"""

    def __init__(self, *args,
                 vars: dict = {},  # collections of application hyperparameters including model parameters.
                 **kwargs):
        # Collect the indexWanted.
        self.indexWanted = [vars[f'eleven_v{i}'].get() for i in range(11)]
        assert self.indexWanted is not None, 'Need to provided the eleven group!'

        # Generate all datasets for every target index in the indexWanted list.
        self.dataset_list = rrd.dataset_for_triple_dicer(self.indexWanted)

        # Initial the three models


    def training_prediction(self):
        """Function to train and predict the results of the three models"""

        # Simple Fully Connected Network
        # encoder and decoder framework
        """
        encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        """

        MODEL_PATH = "models/lstmneutralnetwork/"
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=3,
            verbose=False,
            mode="min"
        )
        model = LSTM_Neutral_Network(learning_rate=1e-3)
        Trainer = L.Trainer(
            default_root_dir=MODEL_PATH,
            accelerator="gpu",
            strategy="ddp",
            callbacks=[early_stop_callback]
        )

        # for i in tqdm(range(len(self.indexWanted)), ncols=100, desc="Simple Fully Connected Network", colour="blue"):
        seed = torch.Generator().manual_seed(666)
        for i in range(len(self.indexWanted)):
            ind = self.indexWanted[i]
            train_dataset = self.dataset_list[i]

            # Split the dataset random into train and validation set and prepare the prediction set
            train_set_size = int(len(train_dataset) * 0.8)
            valid_set_size = len(train_dataset) - train_set_size
            train_set, valid_set = random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)
            pred_set = train_dataset[-readit.TDM_BATCH_SIZE:len(train_dataset)]

            if i ==0:
                train_datasets = train_set
                valid_datasets = valid_set
            else:
                train_datasets = train_datasets + train_set
                valid_datasets = valid_datasets + valid_set

        train_loader = DataLoader(train_datasets, batch_size=readit.TDM_BATCH_SIZE, shuffle=False, drop_last=True)
        valid_loader = DataLoader(valid_datasets, batch_size=readit.TDM_BATCH_SIZE, shuffle=False, drop_last=True)

        # Run learning rate finder
        lr_finder = Trainer.tuner.lr_find(model, min_lr=1e-04, max_lr=1, num_training=30,
                                          train_dataloaders=train_loader,
                                          val_dataloaders=valid_loader
        )

        Trainer.fit(
            model,
            ckpt_path="last",
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader)

        # use the model to predict
        # encoder = model.encoder
        # disable randomness, dropout, etc.
        # encoder.eval()
        # pred_feature, _ = pred_set
        # pred_results = encoder(pred_feature)

        return 1
