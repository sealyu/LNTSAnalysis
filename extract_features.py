import pandas as pd
import numpy as np
from datetime import date

"""
dataset split:
    (date_received)  05-11 ~ 11-12                            
           dateset3: 10-01~11-12 (113640),features3 from 08-16~09-30  (off_test)
           dateset2: 08-16~09-30 (258446),features2 from 07-01~08-15  
           dateset1: 07-01~08-15 (138303),features1 from 05-11~06-30        

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
user_info_data.gender = user_info_data.gender.replace(np.nan,2)
user_info_data.age_range = user_info_data.age_range.replace(np.nan0)
test_Data = pd.read_csv('./data/test_format1.csv')
train_data = pd.read_csv('./data/train_format1.csv')

dataset3 = user_log_data[user_log_data.time_stamp >= 1001]
dataset2 = user_log_data[(user_log_data.time_stamp >= 816) & (user_log_data.time_stamp <= 930)]
feature3 = dataset2
dataset1 = user_log_data[(user_log_data.time_stamp >= 701) & (user_log_data.time_stamp <= 815)]
feature2 = dataset1
feature1 = user_log_data[user_log_data.time_stamp <= 630]

############# merchant related feature   #############
"""
1.merchant related: 
      total_behavior_count,total_sell_count,total_click_count,total_addCart_count,total_addFav_count
      transfer_rate = total_sell_count/total_click_count
"""

# for dataset3
merchant3 = feature3[['seller_id','cat_id', 'brand_id', 'action_type', 'time_stamp']]

t = merchant3[['seller_id']]
t.drop_duplicates(inplace=True)

t1 = merchant3[['seller_id']]
t1['total_behavior_count'] = 1
t1 = t1.groupby('seller_id').agg('sum').reset_index()

t2 = merchant3[(merchant3.action_type == 2)][['seller_id']]
t2['total_sell_count'] = 1
t2 = t2.groupby('seller_id').agg('sum').reset_index()

t3 = merchant3[(merchant3.action_type == 0)][['seller_id']]
t3['total_click_count'] = 1
t3 = t3.groupby('seller_id').agg('sum').reset_index()

t4 = merchant3[(merchant3.action_type == 1)][['seller_id']]
t4['total_addCart_count'] = 1
t4 = t4.groupby('seller_id').agg('sum').reset_index()

t5 = merchant3[(merchant3.action_type == 3)][['seller_id']]
t5['total_addFav_count'] = 1
t5 = t5.groupby('seller_id').agg('sum').reset_index()

merchant3_feature = pd.merge(t, t1, on='seller_id', how='left')
merchant3_feature = pd.merge(merchant3_feature, t2, on='seller_id', how='left')
merchant3_feature = pd.merge(merchant3_feature, t3, on='seller_id', how='left')
merchant3_feature = pd.merge(merchant3_feature, t4, on='seller_id', how='left')
merchant3_feature = pd.merge(merchant3_feature, t5, on='seller_id', how='left')
merchant3_feature.total_behavior_count = merchant3_feature.total_behavior_count.replace(np.nan,0)
merchant3_feature.total_sell_count = merchant3_feature.total_sell_count.replace(np.nan,0)
merchant3_feature.total_click_count = merchant3_feature.total_click_count.replace(np.nan,0)
merchant3_feature.total_addCart_count = merchant3_feature.total_addCart_count.replace(np.nan,0)
merchant3_feature.total_addFav_count = merchant3_feature.total_addFav_count.replace(np.nan,0)
merchant3_feature['transfer_rate'] = merchant3_feature.total_sell_count.astype('float') / merchant3_feature.total_click_count
merchant3_feature.to_csv('data/feature/merchant3_feature.csv', index=None)

# for dataset2
merchant2 = feature2[['seller_id','cat_id', 'brand_id', 'action_type', 'time_stamp']]

t = merchant2[['seller_id']]
t.drop_duplicates(inplace=True)

t1 = merchant2[['seller_id']]
t1['total_behavior_count'] = 1
t1 = t1.groupby('seller_id').agg('sum').reset_index()

t2 = merchant2[(merchant2.action_type == 2)][['seller_id']]
t2['total_sell_count'] = 1
t2 = t2.groupby('seller_id').agg('sum').reset_index()

t3 = merchant2[(merchant2.action_type == 0)][['seller_id']]
t3['total_click_count'] = 1
t3 = t3.groupby('seller_id').agg('sum').reset_index()

t4 = merchant2[(merchant2.action_type == 1)][['seller_id']]
t4['total_addCart_count'] = 1
t4 = t4.groupby('seller_id').agg('sum').reset_index()

t5 = merchant2[(merchant2.action_type == 3)][['seller_id']]
t5['total_addFav_count'] = 1
t5 = t5.groupby('seller_id').agg('sum').reset_index()

merchant2_feature = pd.merge(t, t1, on='seller_id', how='left')
merchant2_feature = pd.merge(merchant2_feature, t2, on='seller_id', how='left')
merchant2_feature = pd.merge(merchant2_feature, t3, on='seller_id', how='left')
merchant2_feature = pd.merge(merchant2_feature, t4, on='seller_id', how='left')
merchant2_feature = pd.merge(merchant2_feature, t5, on='seller_id', how='left')
merchant2_feature.total_behavior_count = merchant2_feature.total_behavior_count.replace(np.nan,0)
merchant2_feature.total_sell_count = merchant2_feature.total_sell_count.replace(np.nan,0)
merchant2_feature.total_click_count = merchant2_feature.total_click_count.replace(np.nan,0)
merchant2_feature.total_addCart_count = merchant2_feature.total_addCart_count.replace(np.nan,0)
merchant2_feature.total_addFav_count = merchant2_feature.total_addFav_count.replace(np.nan,0)
merchant2_feature['transfer_rate'] = merchant2_feature.total_sell_count.astype('float') / merchant2_feature.total_click_count
merchant2_feature.to_csv('data/feature/merchant2_feature.csv', index=None)

# for dataset1
merchant1 = feature1[['seller_id','cat_id', 'brand_id', 'action_type', 'time_stamp']]

t = merchant1[['seller_id']]
t.drop_duplicates(inplace=True)

t1 = merchant1[['seller_id']]
t1['total_behavior_count'] = 1
t1 = t1.groupby('seller_id').agg('sum').reset_index()

t2 = merchant1[(merchant1.action_type == 2)][['seller_id']]
t2['total_sell_count'] = 1
t2 = t2.groupby('seller_id').agg('sum').reset_index()

t3 = merchant1[(merchant1.action_type == 0)][['seller_id']]
t3['total_click_count'] = 1
t3 = t3.groupby('seller_id').agg('sum').reset_index()

t4 = merchant1[(merchant1.action_type == 1)][['seller_id']]
t4['total_addCart_count'] = 1
t4 = t4.groupby('seller_id').agg('sum').reset_index()

t5 = merchant1[(merchant1.action_type == 3)][['seller_id']]
t5['total_addFav_count'] = 1
t5 = t5.groupby('seller_id').agg('sum').reset_index()

merchant1_feature = pd.merge(t, t1, on='seller_id', how='left')
merchant1_feature = pd.merge(merchant1_feature, t2, on='seller_id', how='left')
merchant1_feature = pd.merge(merchant1_feature, t3, on='seller_id', how='left')
merchant1_feature = pd.merge(merchant1_feature, t4, on='seller_id', how='left')
merchant1_feature = pd.merge(merchant1_feature, t5, on='seller_id', how='left')
merchant1_feature.total_behavior_count = merchant1_feature.total_behavior_count.replace(np.nan,0)
merchant1_feature.total_sell_count = merchant1_feature.total_sell_count.replace(np.nan,0)
merchant1_feature.total_click_count = merchant1_feature.total_click_count.replace(np.nan,0)
merchant1_feature.total_addCart_count = merchant1_feature.total_addCart_count.replace(np.nan,0)
merchant1_feature.total_addFav_count = merchant1_feature.total_addFav_count.replace(np.nan,0)
merchant1_feature['transfer_rate'] = merchant1_feature.total_sell_count.astype('float') / merchant1_feature.total_click_count
merchant1_feature.to_csv('data/feature/merchant1_feature.csv', index=None)


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

# for dataset3
user3 = feature3[['user_id', 'seller_id', 'cat_id', 'brand_id', 'time_stamp', 'action_type']]

t = user3[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user3[['user_id', 'seller_id']]
t1.drop_duplicates(inplace=True)
t1.seller_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'seller_id': 'user_count_merchant'}, inplace=True)

t2 = user3[['user_id', 'time_stamp']]
t3 = t2.groupby('user_id').agg('max').reset_index()
t3.rename(columns={'time_stamp': 'user_last_behavior_time'}, inplace=True)

t4 = user3[(user3.action_type == 2)][['user_id']]
t4['user_total_buy_account'] = 1
t4 = t4.groupby('user_id').agg('sum').reset_index()

t5 = user3[(user3.action_type == 0)][['user_id']]
t5['user_total_click_count'] = 1
t5 = t5.groupby('user_id').agg('sum').reset_index()

t6 = user3[(user3.action_type == 1)][['user_id']]
t6['user_total_addCart_count'] = 1
t6 = t6.groupby('user_id').agg('sum').reset_index()

t7 = user3[(user3.action_type == 3)][['user_id']]
t7['user_total_addFav_count'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

"""
t8 = user3[(user3.action_type == 2)][['user_id', 'time_stamp']]
t8['time_stamp'] = 1
t9 = t8.groupby['user_id'].agg('sum').reset_index()
t9.rename(columns={'action_type': 'total_buy_days'}, inplace=True)

t10 = user3[['user_id', 'time_stamp']]
t10['time_stamp'] = 1
t11 = t10.groupby['user_id'].agg('sum').reset_index()
t11.rename(columns={'action_type': 'total_access_days'}, inplace=True)
"""

user3_feature = pd.merge(t, t1, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t3, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t4, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t5, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t6, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t7, on='user_id', how='left')
#user3_feature = pd.merge(user3_feature, t9, on='user_id', how='left')
#user3_feature = pd.merge(user3_feature, t11, on='user_id', how='left')
user3_feature.user_total_buy_account = user3_feature.user_total_buy_account.replace(np.nan,0)
user3_feature.user_total_click_count = user3_feature.user_total_click_count.replace(np.nan,0)
user3_feature.user_total_addCart_count = user3_feature.user_total_addCart_count.replace(np.nan,0)
user3_feature.user_total_addFav_count = user3_feature.user_total_addFav_count.replace(np.nan,0)
user3_feature['user_transfer_rate'] = user3_feature.user_total_buy_account.astype('float') / user3_feature.user_total_click_count.astype('float')
#user3_feature['user_access_transfer_rate'] = user3_feature.total_buy_days.astype('float') / user3_feature.total_access_days.astype('float')
user3_feature = pd.merge(user3_feature, user_info_data, on='user_id', how='left')
user3_feature.to_csv('data/feature/user3_feature.csv', index=None)

# for dataset2
user2 = feature2[['user_id', 'seller_id', 'cat_id', 'brand_id', 'time_stamp', 'action_type']]

t = user2[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user2[['user_id', 'seller_id']]
t1.drop_duplicates(inplace=True)
t1.seller_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'seller_id': 'user_count_merchant'}, inplace=True)

t2 = user2[['user_id', 'time_stamp']]
t3 = t2.groupby('user_id').agg('max').reset_index()
t3.rename(columns={'time_stamp': 'user_last_behavior_time'}, inplace=True)

t4 = user2[(user2.action_type == 2)][['user_id']]
t4['user_total_buy_account'] = 1
t4 = t4.groupby('user_id').agg('sum').reset_index()

t5 = user2[(user2.action_type == 0)][['user_id']]
t5['user_total_click_count'] = 1
t5 = t5.groupby('user_id').agg('sum').reset_index()

t6 = user2[(user2.action_type == 1)][['user_id']]
t6['user_total_addCart_count'] = 1
t6 = t6.groupby('user_id').agg('sum').reset_index()

t7 = user2[(user2.action_type == 3)][['user_id']]
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
user2_feature = pd.merge(t, t1, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t3, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t4, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t5, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t6, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t7, on='user_id', how='left')
#user2_feature = pd.merge(user2_feature, t9, on='user_id', how='left')
#user2_feature = pd.merge(user2_feature, t11, on='user_id', how='left')
user2_feature.user_total_buy_account = user2_feature.user_total_buy_account.replace(np.nan,0)
user2_feature.user_total_click_count = user2_feature.user_total_click_count.replace(np.nan,0)
user2_feature.user_total_addCart_count = user2_feature.user_total_addCart_count.replace(np.nan,0)
user2_feature.user_total_addFav_count = user2_feature.user_total_addFav_count.replace(np.nan,0)
user2_feature['user_transfer_rate'] = user2_feature.user_total_buy_account.astype('float') / user2_feature.user_total_click_count.astype('float')
#user2_feature['user_access_transfer_rate'] = user2_feature.total_buy_days.astype('float') / user2_feature.total_access_days.astype('float')
user2_feature = pd.merge(user2_feature, user_info_data, on='user_id', how='left')
user2_feature.to_csv('data/feature/user2_feature.csv', index=None)

# for dataset1
user1 = feature1[['user_id', 'seller_id', 'cat_id', 'brand_id', 'time_stamp', 'action_type']]

t = user1[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user1[['user_id', 'seller_id']]
t1.drop_duplicates(inplace=True)
t1.seller_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'seller_id': 'user_count_merchant'}, inplace=True)

t2 = user1[['user_id', 'time_stamp']]
t3 = t2.groupby('user_id').agg('max').reset_index()
t3.rename(columns={'time_stamp': 'user_last_behavior_time'}, inplace=True)

t4 = user1[(user1.action_type == 2)][['user_id']]
t4['user_total_buy_account'] = 1
t4 = t4.groupby('user_id').agg('sum').reset_index()

t5 = user1[(user1.action_type == 0)][['user_id']]
t5['user_total_click_count'] = 1
t5 = t5.groupby('user_id').agg('sum').reset_index()

t6 = user1[(user1.action_type == 1)][['user_id']]
t6['user_total_addCart_count'] = 1
t6 = t6.groupby('user_id').agg('sum').reset_index()

t7 = user1[(user1.action_type == 3)][['user_id']]
t7['user_total_addFav_count'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

"""
t8 = user1[(user1.action_type == 2)][['user_id', 'time_stamp']]
t9 = t8.groupby['user_id'].agg('sum').reset_index()
t9.rename(columns={'action_type': 'total_buy_days'}, inplace=True)

t10 = user1[['user_id', 'time_stamp']]
t11 = t10.groupby['user_id'].agg('sum').reset_index()
t11.rename(columns={'action_type': 'total_access_days'}, inplace=True)
"""
user1_feature = pd.merge(t, t1, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t3, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t4, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t5, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t6, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t7, on='user_id', how='left')
#user1_feature = pd.merge(user1_feature, t9, on='user_id', how='left')
#user1_feature = pd.merge(user1_feature, t11, on='user_id', how='left')
user1_feature.user_total_buy_account = user1_feature.user_total_buy_account.replace(np.nan,0)
user1_feature.user_total_click_count = user1_feature.user_total_click_count.replace(np.nan,0)
user1_feature.user_total_addCart_count = user1_feature.user_total_addCart_count.replace(np.nan,0)
user1_feature.user_total_addFav_count = user1_feature.user_total_addFav_count.replace(np.nan,0)
user1_feature['user_transfer_rate'] = user1_feature.user_total_buy_account.astype('float') / user1_feature.user_total_click_count.astype('float')
#user1_feature['user_access_transfer_rate'] = user1_feature.total_buy_days.astype('float') / user1_feature.total_access_days.astype('float')
user1_feature = pd.merge(user1_feature, user_info_data, on='user_id', how='left')
user1_feature.to_csv('data/feature/user1_feature.csv', index=None)

##################  user_merchant related feature #########################

"""
3.user_merchant:
      user_merchant_buy_total, user_merchant_any, times_user_buy_merchant_before. 
"""
# for dataset3
all_user_merchant = feature3[['user_id', 'seller_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature3[(feature3.action_type == 2)][['user_id', 'seller_id', 'time_stamp']]
t = t[['user_id', 'seller_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id', 'seller_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature3[['user_id', 'seller_id', 'time_stamp']]
t1 = t1[['user_id', 'seller_id']]
t1['user_merchant_any'] = 1
t1 = t1.groupby(['user_id', 'seller_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

user_merchant3 = pd.merge(all_user_merchant, t, on=['user_id', 'seller_id'], how='left')
user_merchant3 = pd.merge(user_merchant3, t1, on=['user_id', 'seller_id'], how='left')
user_merchant3.user_merchant_buy_total = user_merchant3.user_merchant_buy_total.replace(np.nan,0)
user_merchant3.user_merchant_any = user_merchant3.user_merchant_any.replace(np.nan,0)
user_merchant3['user_merchant_rate'] = user_merchant3.user_merchant_buy_total.astype('float') / user_merchant3.user_merchant_any.astype('float')
user_merchant3.to_csv('data/feature/user_merchant3.csv', index=None)

# for dataset2
all_user_merchant = feature2[['user_id', 'seller_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature2[(feature2.action_type == 2)][['user_id', 'seller_id', 'time_stamp']]
t = t[['user_id', 'seller_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id', 'seller_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature2[['user_id', 'seller_id', 'time_stamp']]
t1 = t1[['user_id', 'seller_id']]
t1['user_merchant_any'] = 1
t1 = t1.groupby(['user_id', 'seller_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

user_merchant2 = pd.merge(all_user_merchant, t, on=['user_id', 'seller_id'], how='left')
user_merchant2 = pd.merge(user_merchant2, t1, on=['user_id', 'seller_id'], how='left')
user_merchant2.user_merchant_buy_total = user_merchant2.user_merchant_buy_total.replace(np.nan,0)
user_merchant2.user_merchant_any = user_merchant2.user_merchant_any.replace(np.nan,0)
user_merchant2['user_merchant_rate'] = user_merchant2.user_merchant_buy_total.astype('float') / user_merchant2.user_merchant_any.astype('float')
user_merchant2.to_csv('data/feature/user_merchant2.csv', index=None)

# for dataset1
all_user_merchant = feature1[['user_id', 'seller_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature1[(feature1.action_type == 2)][['user_id', 'seller_id', 'time_stamp']]
t = t[['user_id', 'seller_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id', 'seller_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature1[['user_id', 'seller_id', 'time_stamp']]
t1 = t1[['user_id', 'seller_id']]
t1['user_merchant_any'] = 1
t1 = t1.groupby(['user_id', 'seller_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

user_merchant1 = pd.merge(all_user_merchant, t, on=['user_id', 'seller_id'], how='left')
user_merchant1 = pd.merge(user_merchant1, t1, on=['user_id', 'seller_id'], how='left')
user_merchant1.user_merchant_buy_total = user_merchant1.user_merchant_buy_total.replace(np.nan,0)
user_merchant1.user_merchant_any = user_merchant1.user_merchant_any.replace(np.nan,0)
user_merchant1['user_merchant_rate'] = user_merchant1.user_merchant_buy_total.astype('float') / user_merchant1.user_merchant_any.astype('float')
user_merchant1.to_csv('data/feature/user_merchant1.csv', index=None)


##################  generate training and testing set ################
user_merchant3 = pd.read_csv('data/feature/user_merchant3.csv')
merchant3 = pd.read_csv('data/feature/merchant3_feature.csv')
user3 = pd.read_csv('data/feature/user3_feature.csv')
dataset3 = pd.merge(user_merchant3, merchant3, on='seller_id', how='left')
dataset3 = pd.merge(dataset3, user3, on='user_id', how='left')
dataset3.drop_duplicates(inplace=True)
print(dataset3.shape)

dataset3.user_merchant_buy_total = dataset3.user_merchant_buy_total.replace(np.nan, 0)
dataset3.user_merchant_any = dataset3.user_merchant_any.replace(np.nan, 0)

dataset3 = dataset3.replace('null', np.nan)
dataset3.to_csv('data/dataset3.csv', index=None)

user_merchant2 = pd.read_csv('data/feature/user_merchant2.csv')
merchant2 = pd.read_csv('data/feature/merchant2_feature.csv')
user2 = pd.read_csv('data/feature/user2_feature.csv')
dataset2 = pd.merge(user_merchant2, merchant2, on='seller_id', how='left')
dataset2 = pd.merge(dataset2, user2, on='user_id', how='left')
# dataset2.rename(columns={'seller_id':'merchant_id'},inplace=True)
# dataset2 = pd.merge(dataset2, train_data,on=['user_id','merchant_id'],how='left')
dataset2.drop_duplicates(inplace=True)
print(dataset2.shape)

dataset2.user_merchant_buy_total = dataset2.user_merchant_buy_total.replace(np.nan, 0)
dataset2['label'] = dataset2.user_merchant_buy_total.apply(lambda x:1 if x>=2 else 0)
dataset2.user_merchant_any = dataset2.user_merchant_any.replace(np.nan, 0)

dataset2 = dataset2.replace('null', np.nan)
dataset2.to_csv('data/dataset2.csv', index=None)

user_merchant1 = pd.read_csv('data/feature/user_merchant1.csv')
merchant1 = pd.read_csv('data/feature/merchant1_feature.csv')
user1 = pd.read_csv('data/feature/user1_feature.csv')
dataset1 = pd.merge(user_merchant2, merchant2, on='seller_id', how='left')
dataset1 = pd.merge(dataset1, user1, on='user_id', how='left')
# dataset1.rename(columns={'seller_id':'merchant_id'},inplace=True)
# dataset1 = pd.merge(dataset1, train_data,on=['user_id','merchant_id'],how='left')
dataset1.drop_duplicates(inplace=True)
print(dataset1.shape)

dataset1.user_merchant_buy_total = dataset1.user_merchant_buy_total.replace(np.nan, 0)
dataset1['label'] = dataset1.user_merchant_buy_total.apply(lambda x:1 if x>=2 else 0)
dataset1.user_merchant_any = dataset1.user_merchant_any.replace(np.nan, 0)

dataset1 = dataset1.replace('null', np.nan)
dataset1.to_csv('data/dataset1.csv', index=None)
