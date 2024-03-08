"""
Here I position all the global variables of the project.
"""
import numpy as np
import pandas as pd
from collections import namedtuple

# Variables for the dataset generation----------------------------------------------------------------------------------
indexWantedList = pd.read_csv('/home/Beta4090/PycharmProjects/ProfoundRNN_ext/indexWantedList.csv', index_col=0)
indexWantedList = indexWantedList.stack()
indexWantedList = indexWantedList.unstack(0)

indexWanted_CU0 = indexWantedList['indexWanted_CU0'].to_list()
indexWanted_RB0 = indexWantedList['indexWanted_RB0'].to_list()
indexWanted_SCM = indexWantedList['indexWanted_SCM'].to_list()
indexWanted_FINANCE = indexWantedList['indexWanted_FINANCE'].to_list()
indexWanted_default = indexWanted_FINANCE
indexList = list(np.unique(indexWanted_CU0 + indexWanted_RB0 + indexWanted_SCM + indexWanted_FINANCE))
indexList.sort()

# Variable for the triple dicer model.
TDM_SEQUENCE_LEN = 26  # stand for 25 trading days which are very close to one month
TDM_SEQUENCE_PAD = 999 # the int used for <pad>
TDM_MAX_ENCODER_LENGTH = 8
TDM_MAX_PREDICTION_LENGTH = 1
TDM_BATCH_SIZE = 4

FeatureConfig = namedtuple(
    "FeatureConfig",
    [
        "target",
        "index_cols",
        "static_categoricals",
        "static_reals",
        "time_varying_known_categoricals",
        "time_varying_known_reals",
        "time_varying_unkown_reals",
        "group_ids"
    ],
)

# Variables for the GUI-------------------------------------------------------------------------------------------------
elevenValues_dict = {
    f"eleven_v{i}": {'type': 'str', 'value': indexWanted_default[i]} for i in range(11)
}
fields = {
    **elevenValues_dict,
    'indexWanted_default': {'type': 'str', 'values': indexWanted_default},
    'triple dicer count threshold': {'type': 'int', 'value': 44},
    'triple dicer count type': {'type': 'bool', 'value': False},
    'triple dicer stop flag': {'type': 'bool', 'value': False},
    'triple dicer predict date': {'type': 'str', 'value': ''},
    'triple dicer textual results': {'type': 'BoundText', 'value': 'Textual results will be here.'}
}
values_fields_list = ['indexWanted_default']