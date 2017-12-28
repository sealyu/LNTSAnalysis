import pandas as pd
import numpy as np
from datetime import date

"""      
1.merchant related: 
      total_behavior_count,total_sell_count,total_click_count
      transfer_rate = total_sell_count/total_click_count,

2.user related: 
      age, gender, last_behavior_time, total_click_count, total_buy_account, total_buy_merchant_count,
      total_access_days,total_buy_days,
      user_transfer_rate = total_buy_account/total_click_count, 
      access_transfer_rate = total_access_days/total_buy_days,
      不在初次访问品牌时进行购买的订单数/总订单数

3.user-merchant related: 
        total_behavior_count, total_click_count, total_buy_count, total_addFavorite_count, total_addCart_count,
        total_click_count/total_behavior_count,total_buy_count/total_behavior_count,
        total_addFavorite_count/total_behavior_count,total_addCart_count/total_behavior_count,
        buy_ratio = total_buy_count/user_total_buy_count_all_merchant,

"""

user_log_data = pd.read_csv('./data/user_log_format1.csv')
user_info_data = pd.read_csv('./data/user_info_format1.csv')
# 2 for unknown
user_info_data.gender = user_info_data.gender.replace(np.nan,2)
user_info_data.age_range = user_info_data.age_range.replace(np.nan,0)
test_Data = pd.read_csv('./data/test_format1.csv')
train_data = pd.read_csv('./data/train_format1.csv')

############# merchant related feature   #############
"""
1.merchant related: 
      total_behavior_count,total_sell_count,total_click_count,total_addCart_count,total_addFav_count
      transfer_rate = total_sell_count/total_click_count
"""
merchant = user_log_data[['seller_id','cat_id', 'brand_id', 'action_type', 'time_stamp']]

t = merchant[['seller_id']]
t.drop_duplicates(inplace=True)

t1 = merchant[['seller_id']]
t1['total_behavior_count'] = 1
t1 = t1.groupby('seller_id').agg('sum').reset_index()

t2 = merchant[(merchant.action_type == 2)][['seller_id']]
t2['total_sell_count'] = 1
t2 = t2.groupby('seller_id').agg('sum').reset_index()

t3 = merchant[(merchant.action_type == 0)][['seller_id']]
t3['total_click_count'] = 1
t3 = t3.groupby('seller_id').agg('sum').reset_index()

t4 = merchant[(merchant.action_type == 1)][['seller_id']]
t4['total_addCart_count'] = 1
t4 = t4.groupby('seller_id').agg('sum').reset_index()

t5 = merchant[(merchant.action_type == 3)][['seller_id']]
t5['total_addFav_count'] = 1
t5 = t5.groupby('seller_id').agg('sum').reset_index()

merchant_feature = pd.merge(t, t1, on='seller_id', how='left')
merchant_feature = pd.merge(merchant_feature, t2, on='seller_id', how='left')
merchant_feature = pd.merge(merchant_feature, t3, on='seller_id', how='left')
merchant_feature = pd.merge(merchant_feature, t4, on='seller_id', how='left')
merchant_feature = pd.merge(merchant_feature, t5, on='seller_id', how='left')
merchant_feature.total_behavior_count = merchant_feature.total_behavior_count.replace(np.nan,0)
merchant_feature.total_sell_count = merchant_feature.total_sell_count.replace(np.nan,0)
merchant_feature.total_click_count = merchant_feature.total_click_count.replace(np.nan,0)
merchant_feature.total_addCart_count = merchant_feature.total_addCart_count.replace(np.nan,0)
merchant_feature.total_addFav_count = merchant_feature.total_addFav_count.replace(np.nan,0)
merchant_feature['transfer_rate'] = merchant_feature.total_sell_count.astype('float') / merchant_feature.total_click_count
merchant_feature.transfer_rate = merchant_feature.transfer_rate.replace(np.nan,0)
merchant_feature.to_csv('data/feature/merchant_feature.csv', index=None)

############# user related feature   #############
"""
2.user related: 
      user_count_merchant,age, gender, user_last_behavior_time, user_total_behavior_count, user_total_click_count, 
      user_total_addCart_count,user_total_buy_account,user_total_addFav_count,
      total_access_days,total_buy_days,
      user_transfer_rate = total_buy_account/total_click_count, 
      user_access_transfer_rate = total_buy_days/total_access_days,
      不在初次访问品牌时进行购买的订单数/总订单数
"""

user = user_log_data[['user_id', 'seller_id', 'cat_id', 'brand_id', 'time_stamp', 'action_type']]

t = user[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user[['user_id', 'seller_id']]
t1.drop_duplicates(inplace=True)
t1.seller_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'seller_id': 'user_count_merchant'}, inplace=True)

t2 = user[['user_id', 'time_stamp']]
t3 = t2.groupby('user_id').agg('max').reset_index()
t3.rename(columns={'time_stamp': 'user_last_behavior_time'}, inplace=True)

t4 = user[(user.action_type == 2)][['user_id']]
t4['user_total_buy_account'] = 1
t4 = t4.groupby('user_id').agg('sum').reset_index()

t5 = user[(user.action_type == 0)][['user_id']]
t5['user_total_click_count'] = 1
t5 = t5.groupby('user_id').agg('sum').reset_index()

t6 = user[(user.action_type == 1)][['user_id']]
t6['user_total_addCart_count'] = 1
t6 = t6.groupby('user_id').agg('sum').reset_index()

t7 = user[(user.action_type == 3)][['user_id']]
t7['user_total_addFav_count'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

"""
t8 = user2[(user2.action_type == 2)][['user_id', 'time_stamp']]
t9 = t8.groupby['user_id'].agg('sum').reset_index()
t9.rename(columns={'action_type': 'total_buy_days'}, inplace=True)

t10 = user2[['user_id', 'time_stamp']]
t11 = t10.groupby['user_id'].agg('sum').reset_index()
t11.rename(columns={'action_type': 'total_access_days'}, inplace=True)
"""
user_feature = pd.merge(t, t1, on='user_id', how='left')
user_feature = pd.merge(user_feature, t3, on='user_id', how='left')
user_feature = pd.merge(user_feature, t4, on='user_id', how='left')
user_feature = pd.merge(user_feature, t5, on='user_id', how='left')
user_feature = pd.merge(user_feature, t6, on='user_id', how='left')
user_feature = pd.merge(user_feature, t7, on='user_id', how='left')
user_feature.user_count_merchant = user_feature.user_count_merchant.replace(np.nan,0)
user_feature.user_last_behavior_time = user_feature.user_last_behavior_time.replace(np.nan,0)
user_feature.user_total_buy_account = user_feature.user_total_buy_account.replace(np.nan,0)
user_feature.user_total_click_count = user_feature.user_total_click_count.replace(np.nan,0)
user_feature.user_total_addCart_count = user_feature.user_total_addCart_count.replace(np.nan,0)
user_feature.user_total_addFav_count = user_feature.user_total_addFav_count.replace(np.nan,0)
user_feature['user_transfer_rate'] = user_feature.user_total_buy_account.astype('float') / user_feature.user_total_click_count.astype('float')
user_feature.user_transfer_rate = user_feature.user_transfer_rate.replace([np.inf, -np.inf],0)
user_feature = pd.merge(user_feature, user_info_data, on='user_id', how='left')
user_feature.to_csv('data/feature/user_feature.csv', index=None)

##################  user_merchant related feature #########################

"""
3.user_merchant:
      user_merchant_buy_total, user_merchant_any, times_user_buy_merchant_before. 
"""

all_user_merchant = user_log_data[['user_id', 'seller_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = user_log_data[(user_log_data.action_type == 2)][['user_id', 'seller_id', 'time_stamp']]
t = t[['user_id', 'seller_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id', 'seller_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = user_log_data[['user_id', 'seller_id', 'time_stamp']]
t1 = t1[['user_id', 'seller_id']]
t1['user_merchant_any'] = 1
t1 = t1.groupby(['user_id', 'seller_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

user_merchant = pd.merge(all_user_merchant, t, on=['user_id', 'seller_id'], how='left')
user_merchant = pd.merge(user_merchant, t1, on=['user_id', 'seller_id'], how='left')
user_merchant.user_merchant_buy_total = user_merchant.user_merchant_buy_total.replace(np.nan,0)
user_merchant.user_merchant_any = user_merchant.user_merchant_any.replace(np.nan,0)
user_merchant['user_merchant_rate'] = user_merchant.user_merchant_buy_total.astype('float') / user_merchant.user_merchant_any.astype('float')
user_merchant.to_csv('data/feature/user_merchant.csv', index=None)


##################  generate training and testing set ################

user_merchant = pd.read_csv('data/feature/user_merchant.csv')
merchant = pd.read_csv('data/feature/merchant_feature.csv')
user = pd.read_csv('data/feature/user_feature.csv')
dataset = pd.merge(user_merchant, merchant, on='seller_id', how='left')
dataset = pd.merge(dataset, user, on='user_id', how='left')
dataset.drop_duplicates(inplace=True)

dataset.user_merchant_buy_total = dataset.user_merchant_buy_total.replace(np.nan, 0)
dataset['label'] = dataset.user_merchant_buy_total.apply(lambda x:1 if x>=2 else 0)
dataset.user_merchant_any = dataset.user_merchant_any.replace(np.nan, 0)
print(dataset.shape)

dataset2 = dataset.replace('null', np.nan)
dataset2.fillna(0)
dataset2.to_csv('data/feature_set.csv', index=None,float_format='%.2f')

