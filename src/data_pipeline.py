import warnings
warnings.filterwarnings('ignore')
import os     # for interacting with directory
import numpy as np     # for calculation
import pandas as pd     # for manipulating DataFrame
import yaml     # for interacting with config.yaml  
import util as util     # import common function
from sklearn.model_selection import train_test_split     # for splitting train and test data

def populate_raw_data(directory:str) -> pd.DataFrame:
    raw_data = pd.DataFrame()
    for well_data in os.listdir(directory):
        raw_data = pd.concat([raw_data,pd.read_csv(directory+well_data)],
                             axis=0,
                             ignore_index=True)
    return raw_data

def split_train_test_data(raw_data:pd.DataFrame, output:'str'='Facies') -> pd.DataFrame:
    X=raw_data.drop(columns=output, axis=1)
    y=raw_data[output]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=123, stratify=y)
    print(f"Train data consist of X_train : {X_train.shape[0]} row and y_train : {y_train.shape[0]} row")
    print(f"Test data consist of X_test : {X_test.shape[0]} row and y_test : {y_test.shape[0]} row")

    return X_train, y_train, X_test, y_test

def impute_PE_data_train(X_data:pd.DataFrame, y_data:pd.DataFrame) -> pd.DataFrame:
    if X_data["PE"].isnull().sum() == 0:
        pass
    else :
        data = pd.concat([X_data, y_data], axis=1)
        PE_mean_data = data.loc[~(data["PE"].isnull()),["Facies","PE"]].groupby("Facies").agg('mean')
        PE_nan_data = data.loc[data["PE"].isnull(),["Facies","PE"]]
        facies_labels = data["Facies"].unique().tolist()
        for facies in facies_labels:
            index_PE_nan_by_categories = PE_nan_data[PE_nan_data['Facies']==facies].index.to_list()    
            data.loc[index_PE_nan_by_categories,'PE'] = PE_mean_data.loc[facies].values[0]
        X_data = data.drop(columns='Facies', axis=1)
        y_data = data['Facies']
    print(f"Train data has been cleaned")
    return X_data, y_data

def impute_PE_data_test(X_data:pd.DataFrame, y_data:pd.DataFrame, X_imputer:pd.DataFrame, y_imputer:pd.DataFrame) -> pd.DataFrame:
    if X_data["PE"].isnull().sum() == 0:
        pass
    else :
        data = pd.concat([X_data, y_data], axis=1)
        imputer_data = pd.concat([X_imputer, y_imputer], axis=1)
        PE_mean_data = imputer_data.loc[~(imputer_data["PE"].isnull()),["Facies","PE"]].groupby("Facies").agg('mean')
        PE_nan_data = data.loc[data["PE"].isnull(),["Facies","PE"]]
        facies_labels = data["Facies"].unique().tolist()
        for facies in facies_labels:
            index_PE_nan_by_categories = PE_nan_data[PE_nan_data['Facies']==facies].index.to_list()    
            data.loc[index_PE_nan_by_categories,'PE'] = PE_mean_data.loc[facies].values[0]
        X_data = data.drop(columns='Facies', axis=1)
        y_data = data['Facies']
    print(f"Test data has been cleaned")
    return X_data, y_data


if __name__ == "__main__":
    config = util.load_config()
    raw_data = populate_raw_data(config["raw_dataset_dir"])
    util.dump_pickle(raw_data,config["raw_data_path"])     # dump pickle for raw data
    X_train_unclean, y_train_unclean, X_test_unclean, y_test_unclean = split_train_test_data(raw_data = raw_data, output='Facies')     # split train and test data with same proportion of category
    X_train_clean, y_train_clean = impute_PE_data_train(X_data=X_train_unclean, y_data=y_train_unclean)     # impute train data 
    X_test_clean, y_test_clean = impute_PE_data_test(X_data=X_test_unclean, y_data=y_test_unclean,     # impute test data using data from train data
                                                 X_imputer=X_train_unclean, y_imputer=y_train_unclean)

    # dump pickle for clean data                                            
    util.dump_pickle(data=X_train_clean, file_path=config['data_train_path'][0])
    util.dump_pickle(data=y_train_clean, file_path=config['data_train_path'][1])
    util.dump_pickle(data=X_test_clean, file_path=config['data_test_path'][0])
    util.dump_pickle(data=y_test_clean, file_path=config['data_test_path'][1])