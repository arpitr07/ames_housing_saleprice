#Dependencies
import numpy as np
import pandas as pd
from datetime import datetime
pd.set_option('display.max_columns', None)

from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")

#load train and test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#train and test data size
print("Train set size:", train.shape)
print("Test set size:", test.shape)

#Start data processing
print('START data processing', datetime.now(), )

#store unique IDs for each of the house in train and test data sets
train_ID = train['Id']
test_ID = test['Id']

#Drop the 'Id' column from train & test as IDs are not predictor variables
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

#Remove records with outliers from training data
train = train[(train.GrLivArea < 4500)&(train.SalePrice < 200000)]

#reset the index of training data
train.reset_index(drop=True, inplace=True)

#log-transform the target varaible for it to have closer to normal distribution
train["SalePrice"] = np.log1p(train["SalePrice"])

#store the 'SalePrice' in 'y'
y = train.SalePrice.reset_index(drop=True)

#separate the predictor features from the target variable
train_features = train.drop(['SalePrice'], axis=1)
test_features = test

#original_predictors: all except 'Id' and 'SalePrice'
original_predictors = list(train_features.columns)

# import joblib for dumping as .pkl
from sklearn.externals import joblib

#dump original_predictors
joblib.dump(original_predictors, 'original_predictors.pkl')
print("Original predictors dumped!")

#feature_engineering: num_to_str
train_features['MSSubClass'] = train_features['MSSubClass'].apply(str)
train_features['YrSold'] = train_features['YrSold'].astype(str)
train_features['MoSold'] = train_features['MoSold'].astype(str)

#feature_engineering: impute_categorical
train_features['Functional'] = train_features['Functional'].fillna('Typ')
train_features['Electrical'] = train_features['Electrical'].fillna("SBrkr")
train_features['KitchenQual'] = train_features['KitchenQual'].fillna("TA")
train_features['Exterior1st'] = train_features['Exterior1st'].fillna(train_features['Exterior1st'].mode()[0])
train_features['Exterior2nd'] = train_features['Exterior2nd'].fillna(train_features['Exterior2nd'].mode()[0])
train_features['SaleType'] = train_features['SaleType'].fillna(train_features['SaleType'].mode()[0])
train_features["PoolQC"] = train_features["PoolQC"].fillna("None")

#feature_engineering: impute_garage_bsmt
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    train_features[col] = train_features[col].fillna(0)
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train_features[col] = train_features[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train_features[col] = train_features[col].fillna('None')

#feature_engineering: impute_mszoning
train_features['MSZoning'] = train_features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

#determine the modes for MSZoning, grouped by MSSubClass
MSZoning_modes = dict(train_features.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.mode()[0]))

#dump MSZoning_modes
joblib.dump(MSZoning_modes, 'MSZoning_modes.pkl')
print("MSZoning_modes dumped!")

objects = []                    #list to store column names of type 'object'
for i in train_features.columns:
    if train_features[i].dtype == object:
        objects.append(i)

#dump objects
joblib.dump(objects, 'objects.pkl')
print("Objects dumped!")

#update the dataframe for 'objects'
train_features.update(train_features[objects].fillna('None'))

#feature_engineering: impute_lotfrontage
train_features['LotFrontage'] = train_features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#determine the modes of LotFrontage, grouped by Neighborhood
LotFrontage_modes = dict(train_features.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.mode()[0]))

#dump LotFrontage_modes
joblib.dump(LotFrontage_modes, 'LotFrontage_modes.pkl')
print("LotFrontage_modes dumped!")


numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []                   #list to store column names of type 'numeric'
for i in train_features.columns:
    if train_features[i].dtype in numeric_dtypes:
        numerics.append(i)

#dump numerics
joblib.dump(numerics, 'numerics.pkl')
print("Numerics dumped!")

#update the dataframe for 'numerics'
train_features.update(train_features[numerics].fillna(0))


numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []                  #list to store column names of type 'numeric'
for i in train_features.columns:
    if train_features[i].dtype in numeric_dtypes:
        numerics2.append(i)

#determine the skewness of each numeric Series in the DataFrame
skew_features = train_features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

#separate the ones with high skewness
high_skew = skew_features[skew_features > 0.5]

#list of numeric predictors with high skewness
skew_index = list(high_skew.index)

#dump skew_index
joblib.dump(skew_index, 'skew_index.pkl')
print("Skew_Index dumped!")

lmbdas = []                  #list to store the lmbda values for high_skew predictors
for i in skew_index:
    train_features[i] = boxcox1p(train_features[i], boxcox_normmax(train_features[i] + 1))
    lmbdas.append(boxcox_normmax(train_features[i]+1))

#dump lmbdas
joblib.dump(lmbdas, 'lmbdas.pkl')
print("Lmbdas dumped!")

#drop irrelevant features
train_features = train_features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

#add new features
train_features['YrBltAndRemod']=train_features['YearBuilt']+train_features['YearRemodAdd']
train_features['TotalSF']=train_features['TotalBsmtSF'] + train_features['1stFlrSF'] + train_features['2ndFlrSF']
train_features['Total_sqr_footage'] = (train_features['BsmtFinSF1'] + train_features['BsmtFinSF2'] +
                                 train_features['1stFlrSF'] + train_features['2ndFlrSF'])
train_features['Total_Bathrooms'] = (train_features['FullBath'] + (0.5 * train_features['HalfBath']) +
                               train_features['BsmtFullBath'] + (0.5 * train_features['BsmtHalfBath']))
train_features['Total_porch_sf'] = (train_features['OpenPorchSF'] + train_features['3SsnPorch'] +
                              train_features['EnclosedPorch'] + train_features['ScreenPorch'] +
                              train_features['WoodDeckSF'])
train_features['haspool'] = train_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
train_features['has2ndfloor'] = train_features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
train_features['hasgarage'] = train_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
train_features['hasbsmt'] = train_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
train_features['hasfireplace'] = train_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

print(train_features.shape)

#one_hot encode the DataFrame
final_features = pd.get_dummies(train_features).reset_index(drop=True)
print(final_features.shape)
print(*list(final_features), sep = '\n')

X = final_features.iloc[:len(y), :]

print('X', X.shape, 'y', y.shape)

'''
overfit = []
overfit.append('MSZoning_C (all)')

joblib.dump(overfit, 'overfit.pkl')
print("Overfit dumped!")

X = X.drop(overfit, axis=1).copy()
X_sub = X_sub.drop(overfit, axis=1).copy()
'''

print(type(X))
print('SalePrice' in list(X))

print('X', X.shape, 'y', y.shape)

#dump model_columns i.e. list of columns in X
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
print("\n")

print('START ML', datetime.now(), )

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y,
                                    scoring="neg_mean_squared_error",
                                    cv=kfolds))
    return (rmse)


alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

ridge = make_pipeline(RobustScaler(),
                      RidgeCV(alphas=alphas_alt, cv=kfolds,))

lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas=alphas2,
                              random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(),
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas,
                                        cv=kfolds, random_state=42, l1_ratio=e_l1ratio))

svr = make_pipeline(RobustScaler(),
                      SVR(C= 20, epsilon= 0.008, gamma=0.0003,))


gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =42)


lightgbm = LGBMRegressor(objective='regression',
                                       num_leaves=4,
                                       learning_rate=0.01,
                                       n_estimators=5000,
                                       max_bin=200,
                                       bagging_fraction=0.75,
                                       bagging_freq=5,
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       #min_data_in_leaf=2,
                                       #min_sum_hessian_in_leaf=11
                                       )


xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006, random_state=42)

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,
                                            gbr, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)


print('TEST score on CV')

score = cv_rmse(ridge)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(elasticnet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(svr)
print("SVR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lightgbm)
print("Lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(gbr)
print("GradientBoosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(xgboost)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )


print('START Fit')
print(datetime.now(), 'StackingCVRegressor')
stack_gen_model = stack_gen.fit(np.array(X), np.array(y))
print(datetime.now(), 'elasticnet')
elastic_model_full_data = elasticnet.fit(X, y)
print(datetime.now(), 'lasso')
lasso_model_full_data = lasso.fit(X, y)
print(datetime.now(), 'ridge')
ridge_model_full_data = ridge.fit(X, y)
print(datetime.now(), 'svr')
svr_model_full_data = svr.fit(X, y)
print(datetime.now(), 'GradientBoosting')
gbr_model_full_data = gbr.fit(X, y)
print(datetime.now(), 'xgboost')
xgb_model_full_data = xgboost.fit(X, y)
print(datetime.now(), 'lightgbm')
lgb_model_full_data = lightgbm.fit(X, y)

#dump xgb_model_full_data
xgb_model_full_data = xgboost.fit(X, y)
joblib.dump(xgb_model_full_data, 'model.pkl')
print("Model dumped!")

def blend_models_predict(X):
    return ((0.0 * elastic_model_full_data.predict(X)) + \
            (0.0 * lasso_model_full_data.predict(X)) + \
            (0.0 * ridge_model_full_data.predict(X)) + \
            (0.0 * svr_model_full_data.predict(X)) + \
            (0.0 * gbr_model_full_data.predict(X)) + \
            (1.0* xgb_model_full_data.predict(X)) + \
            (0.0 * lgb_model_full_data.predict(X)) + \
            (0.0 * stack_gen_model.predict(np.array(X))))

print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))

'''
def blend_models_predict(X):
    return ((0.1 * elastic_model_full_data.predict(X)) + \
            (0.1 * lasso_model_full_data.predict(X)) + \
            (0.1 * ridge_model_full_data.predict(X)) + \
            (0.1 * svr_model_full_data.predict(X)) + \
            (0.1 * gbr_model_full_data.predict(X)) + \
            (0.15 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.25 * stack_gen_model.predict(np.array(X))))

print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))

print('Predict submission', datetime.now(),)
submission = pd.DataFrame({'Id': [], 'SalePrice': []})
submission['Id'] = test_ID.values
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))

q1 = submission['SalePrice'].quantile(0.0042)
q2 = submission['SalePrice'].quantile(0.99)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission1.csv", index=False)
print('Save submission', datetime.now(),)
'''
