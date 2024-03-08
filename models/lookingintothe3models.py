"""
Have to write this script to test every of the 3 models.
"""

"""
Let's start with the simple fully connected model.
"""

# ==============================
# Step 1: prepare the dataloader
# ==============================
import utils.readrawdata as rrd
import torch
from torch.utils.data import DataLoader, random_split
from utils import readit

indexWanted = ['IH00C1', 'IF00C1', 'IC00C1', 'IM00C1', 'TS00C1', 'T00C1', 'TF00C1', 'TL00C1', 'CU0', 'RB0', 'SCM']
dataset_list = rrd.dataset_for_triple_dicer(indexWanted)

# for i in tqdm(range(len(self.indexWanted)), ncols=100, desc="Simple Fully Connected Network", colour="blue"):
seed = torch.Generator().manual_seed(666)
for i in range(len(indexWanted)):
    ind = indexWanted[i]
    train_dataset = dataset_list[i]

    # Split the dataset random into train and validation set and prepare the prediction set
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size
    train_set, valid_set = random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)
    pred_set = train_dataset[-readit.TDM_BATCH_SIZE:len(train_dataset)]

    if i == 0:
        train_datasets = train_set
        valid_datasets = valid_set
    else:
        train_datasets = train_datasets + train_set
        valid_datasets = valid_datasets + valid_set

train_loader = DataLoader(train_datasets, batch_size=readit.TDM_BATCH_SIZE, shuffle=False, drop_last=True)
valid_loader = DataLoader(valid_datasets, batch_size=readit.TDM_BATCH_SIZE, shuffle=False, drop_last=True)

x, y = next(iter(train_loader))
print(x)
print(y)

# ========================
# Step 2: create the model
# ========================
import lightning as L
from torch import nn, optim
import torch.nn.functional as F
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# Simple Fully connected Neutral Network build using lightning
class Simple_Fully_Connected_Neutral_Network(L.LightningModule):
    """ Simply Fully Connected Neutral Network"""
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
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

# =======================
# Step 3: Train the model
# =======================
MODEL_PATH = "models/simplefullyconnectedneutralnetwork/"
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode="min"
)

model = Simple_Fully_Connected_Neutral_Network(learning_rate=1e-3)

Trainer = L.Trainer(
    default_root_dir=MODEL_PATH,
    accelerator="gpu",
    strategy="ddp",
    callbacks=[early_stop_callback]
)

Trainer.fit(
    model,
    ckpt_path="last",
    train_dataloaders=train_loader,
    val_dataloaders=valid_loader)
