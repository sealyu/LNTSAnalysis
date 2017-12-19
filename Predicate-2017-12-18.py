"""
 This is for the 2017-12-18 predicate task.
 Today I start using scikit learn to predicate the repeat buyers using machine learning.
 -- Haibo Yu
"""
import numpy as np
import pandas as pd


orig_data = pd.DataFrame(pd.read_csv('./data/train_format1.csv'))
uid_list = orig_data.user_id.unique()
merchant_list = orig_data.merchant_id.unique()
n_users = orig_data.user_id.unique().shape[0]
n_merchants = orig_data.merchant_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of merchants = ' + str(n_merchants))

# There are movie ids that are larger than num_movie, have to re-index
uid_mapper = {uid_list[i]: i for i in range(0, n_users)}
mid_mapper = {merchant_list[i]: i for i in range(0, n_merchants)}

# 使用scikit-learn库将数据集分割成测试和训练。Cross_validation.train_test_split根据测试样本的比例（test_size），本例中是0.25，来将数据混洗并分割成两个数据集
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(orig_data, test_size=0.25)


# 计算数据集的稀疏度
# sparsity = round(1.0 - len(train_data) / float(n_users * n_merchants), 3)
# print('The sparsity level of Tianchi is ' + str(sparsity*100) + '%')


# 创建uesr-seller矩阵，此处需创建训练和测试两个UI矩阵
train_data_matrix = np.zeros((n_users, n_merchants))
for line in train_data.itertuples():
    train_data_matrix[uid_mapper[line[1]], mid_mapper[line[2]]] = line[3]

test_data_matrix = np.zeros((n_users, n_merchants))
for line in test_data.itertuples():
    test_data_matrix[uid_mapper[line[1]], mid_mapper[line[2]]] = line[3]

# 利用均方根误差进行评估
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import pairwise_distances

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
            [np.abs(similarity).sum(axis=1)]).T
    elif type == 'merchant':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


# user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
# user_prediction = predict(train_data_matrix, user_similarity, type='user')
# print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
# merchant_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
# merchant_prediction = predict(train_data_matrix, merchant_similarity, type='merchant')
# print('Merchant-based CF RMSE: ' + str(rmse(merchant_prediction, test_data_matrix)))
#
# 使用SVD进行矩阵分解
from scipy.sparse.linalg import svds

u, s, vt = svds(train_data_matrix, k=20)
s_diag_matrix = np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

print('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))
