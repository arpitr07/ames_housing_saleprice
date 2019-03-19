import pandas as pd
import numpy as np
from scipy.special import boxcox1p

def num_to_str(query):
    query['MSSubClass'] = query['MSSubClass'].apply(str)
    query['YrSold'] = query['YrSold'].astype(str)
    query['MoSold'] = query['MoSold'].astype(str)
    return query

def impute_categorical(query):
    query['Functional'] = query['Functional'].fillna('Typ')
    query['Electrical'] = query['Electrical'].fillna("SBrkr")
    query['KitchenQual'] = query['KitchenQual'].fillna("TA")
    query['Exterior1st'] = query['Exterior1st'].fillna("VinylSd")
    query['Exterior2nd'] = query['Exterior2nd'].fillna("VinylSd")
    query['SaleType'] = query['SaleType'].fillna("WD")
    query["PoolQC"] = query["PoolQC"].fillna("None")
    return query

def impute_garage_bsmt(query):
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        query[col] = query[col].fillna(0)
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        query[col] = query[col].fillna('None')
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        query[col] = query[col].fillna('None')
    return query

def impute_mszoning(query, MSZoning_modes):
    query['MSZoning'] = MSZoning_modes[query['MSSubClass'].values[0]]
    return query

def update_objects(query, objects):
    query.update(query[objects].fillna('None'))
    return query

def impute_lotfrontage(query, LotFrontage_modes):
    query['LotFrontage'] = LotFrontage_modes[query['Neighborhood'].values[0]]
    return query

def update_numerics(query, numerics):
    query.update(query[numerics].fillna(0))
    return query

def normalize_skewed(query, skew_index, lmbdas):
    for i in range(len(lmbdas)):
        query[i] = boxcox1p(query[skew_index[i]], lmbdas[i])
    return query

def new_features(query):
    query = query.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
    query['YrBltAndRemod']=query['YearBuilt']+query['YearRemodAdd']
    query['TotalSF']=query['TotalBsmtSF'] + query['1stFlrSF'] + query['2ndFlrSF']

    query['Total_sqr_footage'] = (query['BsmtFinSF1'] + query['BsmtFinSF2'] +
                                     query['1stFlrSF'] + query['2ndFlrSF'])

    query['Total_Bathrooms'] = (query['FullBath'] + (0.5 * query['HalfBath']) +
                                   query['BsmtFullBath'] + (0.5 * query['BsmtHalfBath']))

    query['Total_porch_sf'] = (query['OpenPorchSF'] + query['3SsnPorch'] +
                                  query['EnclosedPorch'] + query['ScreenPorch'] +
                                  query['WoodDeckSF'])
    query['haspool'] = query['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    query['has2ndfloor'] = query['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    query['hasgarage'] = query['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    query['hasbsmt'] = query['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    query['hasfireplace'] = query['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    return query

def get_final_df(query):
    final_df = pd.get_dummies(query).reset_index(drop=True)
    return final_df

def drop_overfit(query, overfit):
    final_df = query.drop(overfit, axis = 1).copy()
    return final_df
