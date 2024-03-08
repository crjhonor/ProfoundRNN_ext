""" This script contains functions to preprocess the time series to make the dataset """
import warnings
import numpy as np
import pandas as pd
from pandas.api.types import is_list_like
import calendar
import datetime
from typing import List, Tuple

def replace_w_npnan(x):
    if x == '':
        return np.nan
    else:
        return x

def datetimeProcessing(dataToproc):
    """function to reprocess the DATE format"""

    dataToproc['DATE_'] = ''
    for i in range(len(dataToproc)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataToproc['DATE_'].iloc[i] = dataToproc['DATE'].iloc[i].to_pydatetime().date()
    dataToproc.index = dataToproc['DATE_']
    return dataToproc

def generate_logr(dataset, isDATE=True):
    if isDATE:
        dataset_DATE = dataset['DATE']
        dataset_noDATE = dataset.drop(columns=['DATE'])
    else:
        dataset_noDATE = dataset

    dataset_noDATE_pct_change = dataset_noDATE.pct_change(periods=1)
    dataset_noDATE_pct_change = dataset_noDATE_pct_change.iloc[1:]
    dataset_noDATE_logr = dataset_noDATE_pct_change.applymap(lambda x: np.log(x + 1))

    if isDATE:
        dataset_DATE = pd.DataFrame(dataset_DATE.iloc[1:])
        returnDataset = dataset_DATE.join(dataset_noDATE_logr)
    else:
        returnDataset = dataset_noDATE_logr
    return returnDataset

def create_continuous_time_index(dataset: pd.DataFrame):
    """Function to create continuous time index for pytorch forcasting"""
    dataset['time_idx'] = range(dataset.shape[0])
    added_features = {'time index': ['time_idx']}

    return dataset, added_features

def _get_bussiness_days(current_date: datetime.date):
    last_day = calendar.monthrange(current_date.year, current_date.month)[1]
    rng = pd.date_range(current_date.replace(day=1), periods=last_day, freq='D')
    business_days = pd.bdate_range(rng[0], rng[-1])
    for n, b in enumerate(business_days.date == current_date):
        if b:
            business_day = n
    return business_day

def generateDatefeature(dataset):
    '''
    Day of the working days in particular month can be very important. Thus I am trying to generate these features.
    :param dataset: Input dataset wanted to generate the date feature
    :return: Return dataset with date feature
    '''
    returnDataset = dataset.copy()
    # Adding year, month and day
    returnDataset = returnDataset.set_index("daily_timestamp", drop=False)
    returnDataset['year'] = pd.to_datetime(returnDataset.index).year.values
    returnDataset['month'] = pd.to_datetime(returnDataset.index).month.values
    returnDataset['day'] = pd.to_datetime(returnDataset.index).day.values
    # Adding weekdays and businessdays
    weekdays = []
    businessdays = []
    for i in range(len(returnDataset)):
        weekday = calendar.weekday(returnDataset['year'].values[i], returnDataset['month'].values[i],
                         returnDataset['day'].values[i])
        businessday = _get_bussiness_days(returnDataset.index[i])
        weekdays.append(weekday)
        businessdays.append(businessday)
    returnDataset['weekday'] = weekdays
    returnDataset['businessday'] = businessdays
    added_features = {'date features': ['year', 'month', 'day', 'weekday', 'businessday']}
    return returnDataset, added_features

def add_lags(
        df: pd.DataFrame,
        lags: List[int],
        column: str,
        ts_id: str = None,
) -> Tuple[pd.DataFrame, List]:
    """Create lags for the column provided and adds them as other columns in the provided dataframe

    :param df: The dataframe in which features needed to be created
    :param lags: List of lags to be created
    :param columns: None of the column to be lagged
    :param ts_id: Column name of Unique ID of a time series to be grouped by before applying the lags.
    :return: Returns a tuple of the new dataframe and a list of features which were added
    """
    assert is_list_like(lags), "'lags' should be a list of all required lags"
    assert (
        column in df.columns
    ), "'column' should be a valid column in the provided dataframe"

    if ts_id is None:
        warnings.warn(
            "Assuming just one unique time series in dataset. If there are multiple, provide 'ts_id' argument"
        )
        # Assuming just one unique time series in dataset
        col_dict = {f"{column}_lag_{l}": df[column].shift(l) for l in lags}
    else:
        assert (
            ts_id in df.columns
        ), "'ts_id' should be a valid column in the provided dataframe"
        col_dict = {
            f"{column}_lag_{l}": df.groupby([ts_id])[column].shift(l) for l in lags
        }
    df = df.assign(**col_dict)
    added_feature = {'lags': list(col_dict.keys())}

    # Still have to deal with missing values before return. I think it is better to consider replacing with 0.0 as the
    # log return values of label is somehow stationary.
    df = df.applymap(lambda x: 0.0 if(type(x) == np.float and pd.isna(x)) else x)

    return df, added_feature



