"""
 This is for the 2017-12-18 predicate task.
 Today I start using .
 -- Haibo Yu
"""
import numpy as np
import pandas as pd


train_data = pd.DataFrame(pd.read_csv('./data/train_format1.csv'))
n_users = train_data.user_id.unique().shape[0]
n_sellers = train_data.merchant_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of sellers = ' + str(n_sellers))

# 使用scikit-learn库将数据集分割成测试和训练。Cross_validation.train_test_split根据测试样本的比例（test_size），本例中是0.25，来将数据混洗并分割成两个数据集
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(train_data, test_size=0.25)
print(len(train_data) / float(n_users * n_sellers))

# 计算数据集的稀疏度
sparsity = round(1.0 - len(train_data) / float(n_users * n_sellers), 3)
print('The sparsity level of Tianchi is ' + str(sparsity*100) + '%')

# 创建uesr-seller矩阵，此处需创建训练和测试两个UI矩阵
train_data_matrix = np.zeros((n_users, n_sellers))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

test_data_matrix = np.zeros((n_users, n_sellers))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]


# 使用SVD进行矩阵分解
from scipy.sparse.linalg import svds

u, s, vt = svds(train_data_matrix, k=20)
s_diag_matrix = np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

# 利用均方根误差进行评估
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))

