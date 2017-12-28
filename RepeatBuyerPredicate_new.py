import pandas as pd
import xgboost as xgb
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import Imputer

feature_set = pd.read_csv('data/feature_set.csv')
feature_set.drop_duplicates(inplace=True)

Y = feature_set[['user_id', 'seller_id',"label"]]
# X = feature_set.drop(["label", "user_id", "seller_id"], 1)
X = feature_set.drop(["label"], 1)

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

dataset_preds = y_test.drop(["label"], 1)
dataset_x = X_test.drop(['user_id','seller_id'], axis=1)

print(X_train.shape)

y_train = y_train.drop(['user_id', 'seller_id'], 1)
X_train = X_train.drop(['user_id', 'seller_id'], 1)
train_set = xgb.DMatrix(X_train, label=y_train)
test_set = xgb.DMatrix(X_test)

params = {'booster': 'gbtree',
          'objective': 'rank:pairwise',
          'eval_metric': 'auc',
          'gamma': 0.1,
          'min_child_weight': 1.1,
          'max_depth': 5,
          'lambda': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'eta': 0.01,
          'tree_method': 'exact',
          'seed': 0,
          'nthread': 12
          }

#train on dataset1, evaluate on dataset2
#watchlist = [(dataset1,'train'),(dataset2,'val')]
#model = xgb.train(params,dataset1,num_boost_round=3000,evals=watchlist,early_stopping_rounds=300)

watchlist = [(train_set, 'train')]
model = xgb.train(params, train_set, num_boost_round=300, evals=watchlist)

# 转储模型和特征映射
model.dump_model('dump.raw.txt')

# predict test set
dataset_preds['label'] = model.predict(test_set)
# dataset_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label)
dataset_preds.sort_values(by=['user_id','seller_id', 'label'], inplace=True)
dataset_preds.to_csv("xgb_preds.csv", index=None, header=None)
print(dataset_preds.describe())

# save feature score
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
fs = []
for (key, value) in feature_score:
    fs.append("{0},{1}\n".format(key, value))

with open('xgb_feature_score.csv', 'w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)
