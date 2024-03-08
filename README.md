# Project ProfoundRNN_ext

### Examples
![sample_rawDataset_logr.png](sample_rawDataset_logr.png)

## TODO: Go deeper into the dicinger theory

Forecasting lower probability simulating dicing a dice is implemented by myself previously and proved to be very useful,
specially while I tried to forecast future data using ht given timeseries. Repeatedly fitting and predicting the 
correlated dataset with different models will generate odds every single day and can be used to forecast a probability
I believed should related to the one times ahead.

I name this project to 'Dicinger Pro Max'

# Folders Instructions

- /scripts # storing the main scripts
- /models # stroring models
- /utils # storing utility scripts such as preprocessing and reading dataset, etc.

## GUI framework using ttk package

The GUI framework still is build upon the ttk package, but deeply and fully implemented using advanced widgets, 
separated scripts and deeper inherited classes even mix in classes. With one single startup codes in the root directory 
and a few hyperparameters saving files, the root is quite clean and easy to read.

## Dicingers Menu

Under this menu, there will be several dicinger models. I've renamed the original dicinger as triple dicinger model as 
there are 3 models in the model to dice.

### Triple Dicinger model

- The original dicinger model which has three deep learning neutral network models there for dice.
- 