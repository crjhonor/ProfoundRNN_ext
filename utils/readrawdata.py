"""
Scripts to read all the raw data and build up the dataset. Dataset should be shared with different models, that's why I
write this script to include every process of dataset generation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xlrd
import warnings
import datetime
from utils import readit
from utils.ts_utils import (
    replace_w_npnan,
    datetimeProcessing,
    generate_logr,
    create_continuous_time_index,
    generateDatefeature,
    add_lags,
)
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import MultiNormalizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

"""Part I:
   Read All the data"""

class readingYields:
    """Class to read yields and preprocess them"""
    def __init__(self, yieldsWanted):
        self.dataDirName = "/run/user/1000/gvfs/smb-share:server=crjlambda-pc,share=profoundrnn_data"
        self.yieldsWanted = yieldsWanted
        self.readFilename = Path(self.dataDirName, 'yields.xls')
        self.workSheet = self._readFiles()
        self.returnFeatures = self._generateFeatures()

    def _readFiles(self):
        yieldsWorkbook = xlrd.open_workbook(self.readFilename)
        workSheet = yieldsWorkbook.sheet_by_index(0)
        return workSheet

    def _generateFeatures(self):
        workSheet = self.workSheet
        yieldLambda = 1 # Try to fine tune.
        # # Loading the data.
        yieldsRead = pd.DataFrame(columns=['DATE']).join(pd.DataFrame(columns=[i for i in self.yieldsWanted]))
        for i in yieldsRead.columns:
            if (i == 'DATE'):
                yieldsRead[i] = [pd.Timestamp(dt.value) for dt in workSheet.col(0)[4:-7]]
            elif (i in ['US_10yry', 'GR_10yry', 'IT_10yry', 'AU_10yry', 'JP_10yry']):
                # locate the feature's col number
                for j in range(workSheet.ncols):
                    if (workSheet.row(0)[j].value == i):
                        tmp_x = j
                # Dealing with the data lagging for 1 day if there is any.
                tmp_y = [i.value for i in workSheet.col(tmp_x)[4:-7]]
                if tmp_y[0] == '' and tmp_y[1] != '':  # During workdays except Monday
                    tmp_y[0] = tmp_y[1]
                elif all([tmp_y[0] == '', tmp_y[1] == '', tmp_y[2] == '']):  # On monday
                    tmp_y[0] = tmp_y[3]
                yieldsRead[i] = tmp_y
            else:
                # locate the feature's col number
                for j in range(workSheet.ncols):
                    if workSheet.row(0)[j].value == i:
                        tmp_x = j
                yieldsRead[i] = [i.value for i in workSheet.col(tmp_x)[4:-7]]

        yieldsRead = yieldsRead.applymap(replace_w_npnan)
        yieldsRead = yieldsRead.dropna()
        # # Generate the yield features.
        returnFeatures = pd.DataFrame(columns=['DATE']).join(pd.DataFrame(columns=[i for i in self.yieldsWanted]))
        for i in returnFeatures.columns:
            if (i=='DATE'):
                returnFeatures[i] = yieldsRead[i][:-1]
            else:
                close_t = np.array(yieldsRead[i][:-1])
                close_tsub1 = np.array(yieldsRead[i][1:])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    returnFeatures[i] = [np.log(close_t[j]/close_tsub1[j])*yieldLambda for j in range(len(close_t))]
        return returnFeatures

class getLabels:
    def __init__(self, indexWanted):
        self.dataDirName = "/run/user/1000/gvfs/smb-share:server=crjlambda-pc,share=profoundrnn_data"
        self.indexWanted = indexWanted
        self.readFilename = Path(self.dataDirName, self.indexWanted+'.xls')
        self.workSheet = self._readFiles()
        self.returnLabels = self._generateLabels()

    def _readFiles(self):
        labelWorkbook = xlrd.open_workbook(self.readFilename)
        workSheet = labelWorkbook.sheet_by_index(0)
        # print('Reading LABEL FILE...<DONE!>...')
        return workSheet

    def _generateLabels(self):
        workSheet = self.workSheet
        returnLabels = pd.DataFrame(columns=['DATE', 'LABEL'])
        returnLabels['DATE'] = [xlrd.xldate_as_datetime(dt.value, 0) for dt in workSheet.col(2)[2:-6]]
        # Using daily log return as the labels.
        close_t = [cl.value for cl in workSheet.col(6)[2:-6]]
        close_t1 = [cl.value for cl in workSheet.col(6)[1:-7]]
        logr = [np.log(close_t[i]/close_t1[i]) for i in range(len(close_t))]
        returnLabels['LABEL'] = logr
        return returnLabels

class readingMacanamonthly:
    def __init__(self):
        self.dataDirName = "/run/user/1000/gvfs/smb-share:server=crjlambda-pc,share=profoundrnn_data"
        self.readFilename = Path(self.dataDirName, 'MacAnaMonthly_2.xls')
        self.workSheet = self.readFiles()
        self.returnFeatures = self.generateFeatures()

    def readFiles(self):
        Workbook = xlrd.open_workbook(self.readFilename)
        workSheet = Workbook.sheet_by_index(0)
        return workSheet

    def generateFeatures(self):
        # Transfer the worksheet reading by xlrd to DataFrame.
        workSheet = self.workSheet
        col_names = []
        for j in range(workSheet.ncols):
            if j == 0:
                col_names.append('DATE')
            else:
                col_names.append(workSheet.cell_value(0,j))

        data = np.ones((workSheet.nrows-4, workSheet.ncols))
        data = pd.DataFrame(data)
        for j in range(data.shape[1]):
            for i in range(data.shape[0]):
                try:
                    data.iloc[i, j] = workSheet.cell_value(i+4, j)
                except:
                    print("err occur:", i, j, workSheet.cell_value(i+4, j))
        data.columns = col_names

        data = data.applymap(replace_w_npnan)
        # copying last months' data if np.nan
        for i in range(data.shape[1]):
            if data.isna().iloc[0, i]:
                data.iloc[0, i] = data.iloc[1, i]

        data = data.dropna()
        # # Generate the yield features.
        returnFeatures = data.copy()
        # returnFeatures = pd.DataFrame(columns=['DATE']).join(pd.DataFrame(columns=[i for i in self.yieldsWanted]))
        # for i in returnFeatures.columns:
        #     if (i=='DATE'):
        #         returnFeatures[i] = yieldsRead[i][:-1]
        #     else:
        #         close_t = np.array(yieldsRead[i][:-1])
        #         close_tsub1 = np.array(yieldsRead[i][1:])
        #         returnFeatures[i] = [np.log(close_t[j]/close_tsub1[j])*yieldLambda for j in range(len(close_t))]
        return returnFeatures

class getTradingdata:
    """Getting the trading data features."""

    def __init__(self, indexWanted: list):
        self.indexWanted = indexWanted
        self.dataDirName = "/run/user/1000/gvfs/smb-share:server=crjlambda-pc,share=deeplearn"
        self.returnFeatures = self._returnDataset()

    def _returnDataset(self):
        TD_all_dataset = pd.read_csv(Path(self.dataDirName, 'TD_All.csv'))

        #Currect the name od DATE column
        TD_all_dataset.rename(columns={'Unnamed: 0': 'DATE'}, inplace=True)
        indX = TD_all_dataset.columns.values

        # Dataset of Close
        indX = ['DATE']
        for ind in self.indexWanted:
            indX.append(ind + 'Close')
        datasetClose = TD_all_dataset[indX]
        datasetClose.index = pd.to_datetime(datasetClose['DATE'])

        # Create the log return for the dataset and return it.
        datasetClose_logr = generate_logr(datasetClose, isDATE=True)

        return datasetClose_logr

dataDirName = "/run/user/1000/gvfs/smb-share:server=crjlambda-pc,share=dataforlater"
emailReadfilename = Path(dataDirName, "II_seq2seq_moon2sun_cook_email_feature_forlater.json")
weiboReadfilename = Path(dataDirName, "II_seq2seq_moon2sun_cook_weibo_feature_forlater.json")
emailFeatures_df = pd.read_json(emailReadfilename)
weiboFeatures_df = pd.read_json(weiboReadfilename)

# Get labels and process both the features and labels for deep learning.
dataDirName = "/run/user/1000/gvfs/smb-share:server=crjlambda-pc,share=dailytds"
TD_indexes = pd.read_csv(Path(dataDirName, 'ref_TD.csv'))
TD_yields_indexes = pd.read_csv(Path(dataDirName, 'ref_yields.csv'))
TD_Currency_indexes = pd.read_csv(Path(dataDirName, 'ref_Currency.csv'))

# And generate wanted dataset
indexesAll = TD_indexes.join(TD_Currency_indexes, rsuffix='_Currency')
# indexesAll = indexesAll.join(TD_yields_indexes, rsuffix='_yields')
indexesAll_ind = indexesAll.iloc[0,]

# Get yields
yieldsWanted = ['CN_10yry', 'US_10yry', 'GR_10yry', 'IT_10yry', 'AU_10yry', 'CN_5yry', 'CN_2yry', 'JP_10yry']
ry = readingYields(yieldsWanted=yieldsWanted)
yieldsFeature_df = ry.returnFeatures

# Read macro economic data
mam = readingMacanamonthly()
mamDL_df = mam.returnFeatures

# Build more assistant functions.
def _fnl_generate(
        target_ind: str,
        fnl_label: pd.DataFrame,
        fnl_tradingFeatures: pd.DataFrame,
        fnl_emailFeatures: pd.DataFrame,
        fnl_weiboFeatures: pd.DataFrame,
        fnl_yieldsFeatures: pd.DataFrame,
        mamDL_df: pd.DataFrame
):
        """Function to generate the features and labels"""

        fnl_emailFeatures = datetimeProcessing(fnl_emailFeatures)
        fnl_weiboFeatures = datetimeProcessing(fnl_weiboFeatures)
        fnl_yieldsFeatures = datetimeProcessing(fnl_yieldsFeatures)
        fnl_label = datetimeProcessing(fnl_label)
        mamDL_df.index = mamDL_df['DATE'].apply(lambda x: pd.to_datetime(x).date())

        # join the features and label into one dataframe
        tradingFeatures_order = fnl_label.shape[1] - 1
        fnl_df = fnl_label.join(fnl_tradingFeatures, rsuffix='_other')
        fnl_df = fnl_df.dropna()

        # Drop the replicated target columns
        fnl_df = fnl_df.drop([target_ind+'Close'], axis="columns")

        # join the email features
        emailFeatures_order = fnl_df.shape[1] - 2
        fnl_df = fnl_df.join(fnl_emailFeatures, rsuffix='_other')
        fnl_df = fnl_df.dropna()
        fnl_df = fnl_df.drop(['DATE', 'DATE_other', 'DATE__other'], axis="columns")

        # join the weibo features
        weiboFeatures_order = fnl_df.shape[1]
        fnl_df = fnl_df.join(fnl_weiboFeatures, rsuffix='_weibo')
        fnl_df = fnl_df.dropna()
        fnl_df = fnl_df.drop(['DATE', 'DATE__weibo'], axis="columns")

        # join the yields features
        yieldsFeatures_order = fnl_df.shape[1]
        fnl_df = fnl_df.join(fnl_yieldsFeatures, rsuffix='_other')
        fnl_df = fnl_df.dropna()
        fnl_df = fnl_df.drop(['DATE', 'DATE__other'], axis="columns")

        # Correct the DATE column name
        fnl_df.rename(columns = {'DATE_': 'DATE'}, inplace=True)

        # It is an advantage to use the dataset generation framework earlier in the tsTransformer project, which have
        # been proved to have a good accuracy while forecasting uptodate label.
        # Extract the timestampes
        daily_timestamps = fnl_df['DATE']

        # For te purpose of forecasting T times head, it makes sense to create these labels.
        Ts = [1, 2]
        labels_dict = {}
        label = fnl_df['LABEL'].to_list()
        for t in Ts:
            ls = label[t:]
            [ls.append(itm) for itm in list(np.repeat(label[-1], t))]
            labels_dict.update({f'label_T{t}': ls})

        # Adding trading features
        tradingdata_features_dict = {
            f"tf_{l-tradingFeatures_order}": fnl_df.iloc[:, l] for l in range(tradingFeatures_order, emailFeatures_order)
        }
        yields_features_dict = {
            f"yf_{l-yieldsFeatures_order}": fnl_df.iloc[:, l] for l in range(yieldsFeatures_order, fnl_df.shape[1])
        }
        sentiment_features_dict = {
            f"sf_{l-emailFeatures_order}": fnl_df.iloc[:, l] for l in range(emailFeatures_order, yieldsFeatures_order)
        }

        # To build up pre dataset
        preDataset = pd.DataFrame({
            "target_ind":target_ind,
            "daily_timestamp":daily_timestamps,
            "label":label,
            **labels_dict,
            **tradingdata_features_dict,
            **yields_features_dict,
            **sentiment_features_dict}
        )

        # Add macro analysis data columns to the single target dataset
        preDataset['DATE'] = ''
        for k in range(preDataset.shape[0]):
            preDataset.loc[preDataset.index[k], 'DATE'] \
                = datetime.datetime.strftime(preDataset['daily_timestamp'][k], format="%Y-%m")
        preDataset.index.name = ''
        saved_index = preDataset.index
        preDataset = preDataset.reset_index(drop=True)
        mamDL_df_copy = mamDL_df.copy()
        mamDL_df_copy = mamDL_df_copy.reset_index(drop=True)
        preDataset = preDataset.merge(mamDL_df_copy, on="DATE", how="left")
        # Dealing with missing value, just simple copy the lastest.
        preDataset = preDataset.interpolate(method="linear")
        preDataset.set_index('daily_timestamp', drop=False, inplace=True)
        preDataset.drop(['DATE'], axis='columns', inplace=True)

        # labels can not be considered as features, lags will be added later.
        mamDL_df_copy = mamDL_df_copy.drop(['DATE'], axis='columns')

        added_features = {
            'trading data features': list(tradingdata_features_dict.keys()),
            'yields features': list(yields_features_dict.keys()),
            'sentiment features': list(sentiment_features_dict.keys()),
            'macro features': mamDL_df_copy.columns.to_list()
        }

        return preDataset, added_features

def _feature_engineering(
        dataset: pd.DataFrame,  # the dataset to be engineered
):
    """Function to further engineered the dataset"""

    return_added_features = {}

    # Creating a continuous time index for PyTorch Forecasting
    dataset, added_features = create_continuous_time_index(dataset)
    return_added_features.update(added_features)

    # Generate date features
    dataset, added_features = generateDatefeature(dataset)
    return_added_features.update((added_features))

    # Add lags
    lags = list(range(readit.TDM_SEQUENCE_LEN))  # Using the project variable
    dataset, added_features = add_lags(dataset, lags=lags, column='label', ts_id='target_ind')
    return_added_features.update(added_features)

    return dataset, return_added_features

# The dataset generation functions.
def dataset_for_TFT(
        indexWanted: list = {}
):
    """Generate the dataset for Time Fusion Transformer which is using in the pytorch-forecasting package.
    But the TimeSeriesDataset is somehow not very compatible with lightning and other models."""

    # Additionally, get the trading dataset.
    getdt = getTradingdata(indexWanted=indexWanted)
    tradingFeatures_df = getdt.returnFeatures
    return_dataset_list = []

    for i in tqdm(range(len(indexWanted)), ncols=100, desc="reading dataset", colour="white"):
        ind = indexWanted[i]
        added_features = dict() # added_features to store the names of all the features added here.

        # First step is to generate all log return features.
        gl = getLabels(ind)
        label_df = gl.returnLabels
        preDataset, fnl_added_features = _fnl_generate(
            ind, label_df, tradingFeatures_df, emailFeatures_df, weiboFeatures_df, yieldsFeature_df,  mamDL_df
        )  # fnl_added_features to store the names of the features added here.
        added_features = fnl_added_features

        # Further engineering of the dataset to have more features
        preDataset, engineering_added_features = _feature_engineering(preDataset)
        added_features.update(engineering_added_features)

        # Then we have to put the preDataset into dataloader
        # In the original triple dicer model, I didn't split the dataset into training, testing and validating.
        # Now let's just skip the split here as well.
        # Define the different features
        feat_config = readit.FeatureConfig(
            target=['label_T1', 'label_T2'],  # Predicting one day and two days ahead as the original model does
            index_cols=['target_ind', 'daily_timestampe'],
            static_categoricals=['target_ind'],  # Categoricals which does not change with time
            static_reals=[],  # Reals which does not change with time
            time_varying_known_categoricals= \
                added_features['macro features'] + \
                added_features['date features'],  # Categoricals which change with time
            time_varying_known_reals=added_features['trading data features'] + \
                                     added_features['sentiment features'] + \
                                     added_features['lags'],  # Reals which change with time
            time_varying_unkown_reals=[  # Reals which change with time, but we don't have the future. Like the target
                'label', 'label_T1', 'label_T2'
            ],
            group_ids=[  # Features or list of features which uniquely identifies each entity
                'target_ind'
            ]
        )

        # Converting the categoricals to 'object' dtype
        for itm in feat_config.static_categoricals + feat_config.time_varying_known_categoricals:
            preDataset[itm] = preDataset[itm].astype('object')

        timeseriesDataset = TimeSeriesDataSet(
            preDataset,
            time_idx='time_idx',
            target=feat_config.target,
            group_ids=feat_config.group_ids,
            max_encoder_length=readit.TDM_MAX_ENCODER_LENGTH,
            max_prediction_length=readit.TDM_MAX_PREDICTION_LENGTH,
            static_categoricals=feat_config.static_categoricals,
            static_reals=feat_config.static_reals,
            time_varying_known_categoricals=feat_config.time_varying_known_categoricals,
            time_varying_known_reals=feat_config.time_varying_known_reals,
            time_varying_unknown_reals=[
                'label', 'label_T1', 'label_T2'
            ],
            # # TODO: might need to add a normalizer for the targets
        )

        return_dataset_list.append(timeseriesDataset)

    return return_dataset_list, feat_config

# Re define the dataset class
class triple_dicer_dataset(Dataset):
    """Dataset for the triple dicer model"""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        feature = self.features[item]
        label = self.labels[item]
        # Turn them to tensor
        feature = torch.from_numpy(feature).to(torch.float32)
        label = torch.from_numpy(label).to(torch.float32)
        return feature, label

def dataset_for_triple_dicer(
        indexWanted: list = {}
):
    """Generate the dataset for triple dicer model."""

    # Additionally, get the trading dataset.
    getdt = getTradingdata(indexWanted=indexWanted)
    tradingFeatures_df = getdt.returnFeatures
    return_dataset_list = []

    for i in tqdm(range(len(indexWanted)), ncols=100, desc="reading dataset", colour="white"):
        ind = indexWanted[i]
        added_features = dict() # added_features to store the names of all the features added here.

        # First step is to generate all log return features.
        gl = getLabels(ind)
        label_df = gl.returnLabels
        preDataset, fnl_added_features = _fnl_generate(
            ind, label_df, tradingFeatures_df, emailFeatures_df, weiboFeatures_df, yieldsFeature_df,  mamDL_df
        )  # fnl_added_features to store the names of the features added here.
        added_features = fnl_added_features

        # Further engineering of the dataset to have more features
        preDataset, engineering_added_features = _feature_engineering(preDataset)
        added_features.update(engineering_added_features)

        # Then we have to put the preDataset into dataloader
        # In the original triple dicer model, I didn't split the dataset into training, testing and validating.
        # Now let's just skip the split here as well.
        # Define the different features, the nanmed_tuple feat_config is still useful here.
        feat_config = readit.FeatureConfig(
            target=['label_T1', 'label_T2'],  # Predicting one day and two days ahead as the original model does
            index_cols=['target_ind', 'daily_timestampe'],
            static_categoricals=['target_ind'],  # Categoricals which does not change with time
            static_reals=[],  # Reals which does not change with time
            time_varying_known_categoricals= \
                added_features['macro features'] + \
                added_features['date features'],  # Categoricals which change with time
            time_varying_known_reals=added_features['trading data features'] + \
                                     added_features['sentiment features'] + \
                                     added_features['lags'],  # Reals which change with time
            time_varying_unkown_reals=[  # Reals which change with time, but we don't have the future. Like the target
                'label', 'label_T1', 'label_T2'
            ],
            group_ids=[  # Features or list of features which uniquely identifies each entity
                'target_ind'
            ]
        )

        # Converting the categoricals to 'object' dtype
        # for itm in feat_config.static_categoricals + feat_config.time_varying_known_categoricals:
        #     preDataset[itm] = preDataset[itm].astype('object')

        # from dataframe to dataloader
        x_train = preDataset[feat_config.time_varying_known_reals + \
                             feat_config.time_varying_known_categoricals + ["time_idx"]].copy()
        y_train = preDataset[feat_config.target].copy()
        train_dataset = triple_dicer_dataset(features=x_train.to_numpy(), labels=y_train.to_numpy())
        return_dataset_list.append(train_dataset)

    return return_dataset_list
