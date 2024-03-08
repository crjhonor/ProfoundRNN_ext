import csv
from pathlib import Path
import os
import utils.readit as readit
from decimal import Decimal
from datetime import datetime
import json

class HyperparameterSaving:
  """A model for saving hyperparameters"""

  fields = readit.fields
  values_fields_list = readit.values_fields_list

  def __init__(self):
    filename = 'DPM_hyperparameters.json'
    self.filepath = Path('.') / filename

    # load in saved hyperparameters
    self.load()

  def load(self):
    """Load the hyperparameters from the file"""

    # If the file doesn't exist
    if not self.filepath.exists():
      return

    # Open the file and read the raw hyperparameters
    with open(self.filepath, 'r') as fh:
      raw_hyperparameters = json.load(fh)

    for key in self.fields:
      if key in self.values_fields_list:
        if key in raw_hyperparameters and 'values' in raw_hyperparameters[key]:
          raw_hyperparameter = raw_hyperparameters[key]['values']
          self.fields[key]['values'] = raw_hyperparameter
      else:
        if key in raw_hyperparameters and 'value' in raw_hyperparameters[key]:
          raw_hyperparameter = raw_hyperparameters[key]['value']
          self.fields[key]['value'] = raw_hyperparameter

  def save(self):
    with open(self.filepath, 'w') as fh:
      json.dump(self.fields, fh)

  def set(self, key, value):
    """Set the values in fields"""

    if (
      key in self.fields
            # and type(value).__name__ == self.fields[key]['type']
    ):
      if key in self.values_fields_list:
        self.fields[key]['values'] = value
      else:
        self.fields[key]['value'] = value
    else:
      raise ValueError('Bad key or wrong variable type')