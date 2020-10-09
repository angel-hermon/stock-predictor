import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.base import BaseEstimator,TransformerMixin


Volatility_group_labels = ['Low', 'Medium', 'High']

def calculate_voletility(dataframe):
    dataframe = dataframe.copy()
    dataframe['Volatility'] = dataframe['High'] - dataframe['Low']
    return dataframe


def convert_to_float(dataframe):
    dataframe = dataframe.copy()
    for column in ['Close', 'Volume']:
        dataframe[column] = dataframe[column].str.replace(',', '')
        dataframe[column] = dataframe[column].astype(float)
    return dataframe
        
def convert_date(dataframe):
    dataframe = dataframe.copy()
    dataframe['Date'] = pd.to_datetime(dataframe['Date'], format='%m/%d/%Y')
    return dataframe

def sort_by_date(dataframe):
    dataframe = dataframe.copy()
    dataframe = dataframe.sort_values(by='Date')
    return dataframe

def extract_day_of_week(dataframe):
    dataframe = dataframe.copy()
    dataframe['Day_of_week'] = dataframe['Date'].dt.dayofweek
    return dataframe

def set_volatility_group_label(dataframe):
    dataframe = dataframe.copy()
    dataframe['Volatility_group'] = pd.qcut(dataframe['Volatility'], q=3, labels=Volatility_group_labels)
    return dataframe

def calculate_close_30_moving_avarage(dataframe):
    dataframe = dataframe.copy()
    dataframe['Close_30_moving_avarage'] = dataframe['Close'].rolling(30).mean()
    return dataframe
    
def calculate_close_30_moving_avarage_diff(dataframe):
    dataframe = dataframe.copy()
    dataframe['Close_30_moving_avarage_diff'] = dataframe['Close'] - dataframe['Close_30_moving_avarage']
    return dataframe

def set_is_close_below_30_moving_avarage_diff(dataframe):
    dataframe = dataframe.copy()
    dataframe['Is_close_below_30_moving_avarage'] = np.where(dataframe['Close'] < dataframe['Close_30_moving_avarage'], 1, 0)
    return dataframe

def set_is_close_above_30_moving_avarage_diff(dataframe):
    dataframe = dataframe.copy()
    dataframe['Is_close_above_30_moving_avarage'] = np.where(dataframe['Close'] < dataframe['Close_30_moving_avarage'], 0, 1)
    return dataframe

def calculate_days_close_below_30_moving_avarage(dataframe):
    dataframe = dataframe.copy()
    is_close_below_30_moving_avarage_cumsum = dataframe['Is_close_below_30_moving_avarage'].cumsum()
    dataframe['Days_close_below_30_moving_avarage'] = is_close_below_30_moving_avarage_cumsum.sub(is_close_below_30_moving_avarage_cumsum.mask(dataframe['Is_close_below_30_moving_avarage'] != 0).ffill(), fill_value=0).astype(int)
    return dataframe

def calculate_days_close_above_30_moving_avarage(dataframe):
    dataframe = dataframe.copy()
    is_close_above_30_moving_avarage_cumsum = dataframe['Is_close_above_30_moving_avarage'].cumsum()
    dataframe['Days_close_above_30_moving_avarage'] = is_close_above_30_moving_avarage_cumsum.sub(is_close_above_30_moving_avarage_cumsum.mask(dataframe['Is_close_above_30_moving_avarage'] != 0).ffill(), fill_value=0).astype(int)
    return dataframe

def column_selector(dataframe,cols):
    dataframe = dataframe.copy()
    return dataframe[cols]

def get_transformer_func_list():
    return [(func_tuple[0], FunctionTransformer(func_tuple[1],validate=False)) for func_tuple in [ \
            ('calculate_volatility', calculate_voletility), \
            ('convert_to_float', convert_to_float), \
            ('convert_date', convert_date), \
            ('sort_by_date', sort_by_date), \
            ('extract_day_of_week', extract_day_of_week), \
            ('set_volatility_group_label', set_volatility_group_label), \
            ('calculate_close_30_moving_avarage', calculate_close_30_moving_avarage), \
            ('set_is_close_below_30_moving_avarage_diff', set_is_close_below_30_moving_avarage_diff), \
            ('set_is_close_above_30_moving_avarage_diff', set_is_close_above_30_moving_avarage_diff), \
            ('calculate_days_close_below_30_moving_avarage', calculate_days_close_below_30_moving_avarage), \
            ('calculate_days_close_above_30_moving_avarage', calculate_days_close_above_30_moving_avarage)]]

class CustomTransformerScaler(BaseEstimator, TransformerMixin):
    def __init__(self,method='standard'):
        self.method=method
        self.scaler = None

    def fit(self, X, y=None, **kwargs):
        if (self.method=='standard'):
            self.scaler = StandardScaler()
        elif(self.method=='minmax'):
            self.scaler = MinMaxScaler()

        self.scaler.fit(X)  
        return self


    def transform(self, X, y=None, **kwargs):
        X = X.copy()
        columns_names = X.columns
        scaled_data = self.scaler.transform(X)
        scaled_data_df = pd.DataFrame(scaled_data,columns=columns_names)
        return scaled_data_df
            


    def fit_transform(self, X, y=None, **kwargs):

        self = self.fit(X, y)
        return self.transform(X, y)
