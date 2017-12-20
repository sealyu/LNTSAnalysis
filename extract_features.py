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
test_Data = pd.read_csv('./data/test_format1.csv')
train_data = pd.read_csv('./data/train_format1.csv')

dataset3 = user_log_data[user_log_data.time_stamp >= '1001']
dataset2 = user_log_data[(user_log_data.time_stamp >= '816') & (user_log_data.time_stamp <= '930')]
feature3 = dataset2
dataset1 = user_log_data[(user_log_data.time_stamp >= '701') & (user_log_data.time_stamp <= '815')]
feature2 = dataset1
feature1 = user_log_data[user_log_data.time_stamp <= '630']

############# merchant related feature   #############
"""
1.merchant related: 
      total_behavior_count,total_sell_count,total_click_count
      transfer_rate = total_sell_count/total_click_count,
      
      total_sales. sales_use_coupon.  total_coupon
      coupon_rate = sales_use_coupon/total_sales.  
      transfer_rate = sales_use_coupon/total_coupon. 
      merchant_avg_distance,merchant_min_distance,merchant_max_distance of those use coupon

"""

# for dataset3
merchant3 = feature3[['seller_id', 'brand_id', 'action_type', 'time_stamp']]

t = merchant3[['seller_id']]
t.drop_duplicates(inplace=True)

t1 = merchant3[merchant3.date != 'null'][['merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()

t2 = merchant3[(merchant3.date != 'null') & (merchant3.coupon_id != 'null')][['merchant_id']]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()

t3 = merchant3[merchant3.coupon_id != 'null'][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()

t4 = merchant3[(merchant3.date != 'null') & (merchant3.coupon_id != 'null')][['merchant_id', 'distance']]
t4.replace('null', -1, inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1, np.nan, inplace=True)
t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)

merchant3_feature = pd.merge(t, t1, on='merchant_id', how='left')
merchant3_feature = pd.merge(merchant3_feature, t2, on='merchant_id', how='left')
merchant3_feature = pd.merge(merchant3_feature, t3, on='merchant_id', how='left')
merchant3_feature = pd.merge(merchant3_feature, t5, on='merchant_id', how='left')
merchant3_feature = pd.merge(merchant3_feature, t6, on='merchant_id', how='left')
merchant3_feature = pd.merge(merchant3_feature, t7, on='merchant_id', how='left')
merchant3_feature = pd.merge(merchant3_feature, t8, on='merchant_id', how='left')
merchant3_feature.sales_use_coupon = merchant3_feature.sales_use_coupon.replace(np.nan, 0)  # fillna with 0
merchant3_feature['merchant_coupon_transfer_rate'] = merchant3_feature.sales_use_coupon.astype(
    'float') / merchant3_feature.total_coupon
merchant3_feature['coupon_rate'] = merchant3_feature.sales_use_coupon.astype('float') / merchant3_feature.total_sales
merchant3_feature.total_coupon = merchant3_feature.total_coupon.replace(np.nan, 0)  # fillna with 0
merchant3_feature.to_csv('data/merchant3_feature.csv', index=None)

# for dataset2
merchant2 = feature2[['merchant_id', 'coupon_id', 'distance', 'date_received', 'date']]

t = merchant2[['merchant_id']]
t.drop_duplicates(inplace=True)

t1 = merchant2[merchant2.date != 'null'][['merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()

t2 = merchant2[(merchant2.date != 'null') & (merchant2.coupon_id != 'null')][['merchant_id']]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()

t3 = merchant2[merchant2.coupon_id != 'null'][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()

t4 = merchant2[(merchant2.date != 'null') & (merchant2.coupon_id != 'null')][['merchant_id', 'distance']]
t4.replace('null', -1, inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1, np.nan, inplace=True)
t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)

merchant2_feature = pd.merge(t, t1, on='merchant_id', how='left')
merchant2_feature = pd.merge(merchant2_feature, t2, on='merchant_id', how='left')
merchant2_feature = pd.merge(merchant2_feature, t3, on='merchant_id', how='left')
merchant2_feature = pd.merge(merchant2_feature, t5, on='merchant_id', how='left')
merchant2_feature = pd.merge(merchant2_feature, t6, on='merchant_id', how='left')
merchant2_feature = pd.merge(merchant2_feature, t7, on='merchant_id', how='left')
merchant2_feature = pd.merge(merchant2_feature, t8, on='merchant_id', how='left')
merchant2_feature.sales_use_coupon = merchant2_feature.sales_use_coupon.replace(np.nan, 0)  # fillna with 0
merchant2_feature['merchant_coupon_transfer_rate'] = merchant2_feature.sales_use_coupon.astype(
    'float') / merchant2_feature.total_coupon
merchant2_feature['coupon_rate'] = merchant2_feature.sales_use_coupon.astype('float') / merchant2_feature.total_sales
merchant2_feature.total_coupon = merchant2_feature.total_coupon.replace(np.nan, 0)  # fillna with 0
merchant2_feature.to_csv('data/merchant2_feature.csv', index=None)

# for dataset1
merchant1 = feature1[['merchant_id', 'coupon_id', 'distance', 'date_received', 'date']]

t = merchant1[['merchant_id']]
t.drop_duplicates(inplace=True)

t1 = merchant1[merchant1.date != 'null'][['merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()

t2 = merchant1[(merchant1.date != 'null') & (merchant1.coupon_id != 'null')][['merchant_id']]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()

t3 = merchant1[merchant1.coupon_id != 'null'][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()

t4 = merchant1[(merchant1.date != 'null') & (merchant1.coupon_id != 'null')][['merchant_id', 'distance']]
t4.replace('null', -1, inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1, np.nan, inplace=True)
t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)

merchant1_feature = pd.merge(t, t1, on='merchant_id', how='left')
merchant1_feature = pd.merge(merchant1_feature, t2, on='merchant_id', how='left')
merchant1_feature = pd.merge(merchant1_feature, t3, on='merchant_id', how='left')
merchant1_feature = pd.merge(merchant1_feature, t5, on='merchant_id', how='left')
merchant1_feature = pd.merge(merchant1_feature, t6, on='merchant_id', how='left')
merchant1_feature = pd.merge(merchant1_feature, t7, on='merchant_id', how='left')
merchant1_feature = pd.merge(merchant1_feature, t8, on='merchant_id', how='left')
merchant1_feature.sales_use_coupon = merchant1_feature.sales_use_coupon.replace(np.nan, 0)  # fillna with 0
merchant1_feature['merchant_coupon_transfer_rate'] = merchant1_feature.sales_use_coupon.astype(
    'float') / merchant1_feature.total_coupon
merchant1_feature['coupon_rate'] = merchant1_feature.sales_use_coupon.astype('float') / merchant1_feature.total_sales
merchant1_feature.total_coupon = merchant1_feature.total_coupon.replace(np.nan, 0)  # fillna with 0
merchant1_feature.to_csv('data/merchant1_feature.csv', index=None)

############# user related feature   #############
"""
3.user related: 
      count_merchant. 
      user_avg_distance, user_min_distance,user_max_distance. 
      buy_use_coupon. buy_total. coupon_received.
      buy_use_coupon/coupon_received. 
      buy_use_coupon/buy_total
      user_date_datereceived_gap
"""


def get_user_date_datereceived_gap(s):
    s = s.split(':')
    return (
    date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days


# for dataset3
user3 = feature3[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]

t = user3[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user3[user3.date != 'null'][['user_id', 'merchant_id']]
t1.drop_duplicates(inplace=True)
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

t2 = user3[(user3.date != 'null') & (user3.coupon_id != 'null')][['user_id', 'distance']]
t2.replace('null', -1, inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1, np.nan, inplace=True)
t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

t7 = user3[(user3.date != 'null') & (user3.coupon_id != 'null')][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

t8 = user3[user3.date != 'null'][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()

t9 = user3[user3.coupon_id != 'null'][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()

t10 = user3[(user3.date_received != 'null') & (user3.date != 'null')][['user_id', 'date_received', 'date']]
t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
t10 = t10[['user_id', 'user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

user3_feature = pd.merge(t, t1, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t3, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t4, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t5, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t6, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t7, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t8, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t9, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t11, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t12, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t13, on='user_id', how='left')
user3_feature.count_merchant = user3_feature.count_merchant.replace(np.nan, 0)
user3_feature.buy_use_coupon = user3_feature.buy_use_coupon.replace(np.nan, 0)
user3_feature['buy_use_coupon_rate'] = user3_feature.buy_use_coupon.astype('float') / user3_feature.buy_total.astype(
    'float')
user3_feature['user_coupon_transfer_rate'] = user3_feature.buy_use_coupon.astype(
    'float') / user3_feature.coupon_received.astype('float')
user3_feature.buy_total = user3_feature.buy_total.replace(np.nan, 0)
user3_feature.coupon_received = user3_feature.coupon_received.replace(np.nan, 0)
user3_feature.to_csv('data/user3_feature.csv', index=None)

# for dataset2
user2 = feature2[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]

t = user2[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user2[user2.date != 'null'][['user_id', 'merchant_id']]
t1.drop_duplicates(inplace=True)
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

t2 = user2[(user2.date != 'null') & (user2.coupon_id != 'null')][['user_id', 'distance']]
t2.replace('null', -1, inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1, np.nan, inplace=True)
t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

t7 = user2[(user2.date != 'null') & (user2.coupon_id != 'null')][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

t8 = user2[user2.date != 'null'][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()

t9 = user2[user2.coupon_id != 'null'][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()

t10 = user2[(user2.date_received != 'null') & (user2.date != 'null')][['user_id', 'date_received', 'date']]
t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
t10 = t10[['user_id', 'user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

user2_feature = pd.merge(t, t1, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t3, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t4, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t5, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t6, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t7, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t8, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t9, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t11, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t12, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t13, on='user_id', how='left')
user2_feature.count_merchant = user2_feature.count_merchant.replace(np.nan, 0)
user2_feature.buy_use_coupon = user2_feature.buy_use_coupon.replace(np.nan, 0)
user2_feature['buy_use_coupon_rate'] = user2_feature.buy_use_coupon.astype('float') / user2_feature.buy_total.astype(
    'float')
user2_feature['user_coupon_transfer_rate'] = user2_feature.buy_use_coupon.astype(
    'float') / user2_feature.coupon_received.astype('float')
user2_feature.buy_total = user2_feature.buy_total.replace(np.nan, 0)
user2_feature.coupon_received = user2_feature.coupon_received.replace(np.nan, 0)
user2_feature.to_csv('data/user2_feature.csv', index=None)

# for dataset1
user1 = feature1[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]

t = user1[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user1[user1.date != 'null'][['user_id', 'merchant_id']]
t1.drop_duplicates(inplace=True)
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

t2 = user1[(user1.date != 'null') & (user1.coupon_id != 'null')][['user_id', 'distance']]
t2.replace('null', -1, inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1, np.nan, inplace=True)
t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

t7 = user1[(user1.date != 'null') & (user1.coupon_id != 'null')][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

t8 = user1[user1.date != 'null'][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()

t9 = user1[user1.coupon_id != 'null'][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()

t10 = user1[(user1.date_received != 'null') & (user1.date != 'null')][['user_id', 'date_received', 'date']]
t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
t10 = t10[['user_id', 'user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

user1_feature = pd.merge(t, t1, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t3, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t4, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t5, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t6, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t7, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t8, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t9, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t11, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t12, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t13, on='user_id', how='left')
user1_feature.count_merchant = user1_feature.count_merchant.replace(np.nan, 0)
user1_feature.buy_use_coupon = user1_feature.buy_use_coupon.replace(np.nan, 0)
user1_feature['buy_use_coupon_rate'] = user1_feature.buy_use_coupon.astype('float') / user1_feature.buy_total.astype(
    'float')
user1_feature['user_coupon_transfer_rate'] = user1_feature.buy_use_coupon.astype(
    'float') / user1_feature.coupon_received.astype('float')
user1_feature.buy_total = user1_feature.buy_total.replace(np.nan, 0)
user1_feature.coupon_received = user1_feature.coupon_received.replace(np.nan, 0)
user1_feature.to_csv('data/user1_feature.csv', index=None)

##################  user_merchant related feature #########################

"""
4.user_merchant:
      times_user_buy_merchant_before. 
"""
# for dataset3
all_user_merchant = feature3[['user_id', 'merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature3[['user_id', 'merchant_id', 'date']]
t = t[t.date != 'null'][['user_id', 'merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature3[['user_id', 'merchant_id', 'coupon_id']]
t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature3[['user_id', 'merchant_id', 'date', 'date_received']]
t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][['user_id', 'merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature3[['user_id', 'merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature3[['user_id', 'merchant_id', 'date', 'coupon_id']]
t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][['user_id', 'merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant3 = pd.merge(all_user_merchant, t, on=['user_id', 'merchant_id'], how='left')
user_merchant3 = pd.merge(user_merchant3, t1, on=['user_id', 'merchant_id'], how='left')
user_merchant3 = pd.merge(user_merchant3, t2, on=['user_id', 'merchant_id'], how='left')
user_merchant3 = pd.merge(user_merchant3, t3, on=['user_id', 'merchant_id'], how='left')
user_merchant3 = pd.merge(user_merchant3, t4, on=['user_id', 'merchant_id'], how='left')
user_merchant3.user_merchant_buy_use_coupon = user_merchant3.user_merchant_buy_use_coupon.replace(np.nan, 0)
user_merchant3.user_merchant_buy_common = user_merchant3.user_merchant_buy_common.replace(np.nan, 0)
user_merchant3['user_merchant_coupon_transfer_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype(
    'float') / user_merchant3.user_merchant_received.astype('float')
user_merchant3['user_merchant_coupon_buy_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype(
    'float') / user_merchant3.user_merchant_buy_total.astype('float')
user_merchant3['user_merchant_rate'] = user_merchant3.user_merchant_buy_total.astype(
    'float') / user_merchant3.user_merchant_any.astype('float')
user_merchant3['user_merchant_common_buy_rate'] = user_merchant3.user_merchant_buy_common.astype(
    'float') / user_merchant3.user_merchant_buy_total.astype('float')
user_merchant3.to_csv('data/user_merchant3.csv', index=None)

# for dataset2
all_user_merchant = feature2[['user_id', 'merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature2[['user_id', 'merchant_id', 'date']]
t = t[t.date != 'null'][['user_id', 'merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature2[['user_id', 'merchant_id', 'coupon_id']]
t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature2[['user_id', 'merchant_id', 'date', 'date_received']]
t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][['user_id', 'merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature2[['user_id', 'merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature2[['user_id', 'merchant_id', 'date', 'coupon_id']]
t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][['user_id', 'merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant2 = pd.merge(all_user_merchant, t, on=['user_id', 'merchant_id'], how='left')
user_merchant2 = pd.merge(user_merchant2, t1, on=['user_id', 'merchant_id'], how='left')
user_merchant2 = pd.merge(user_merchant2, t2, on=['user_id', 'merchant_id'], how='left')
user_merchant2 = pd.merge(user_merchant2, t3, on=['user_id', 'merchant_id'], how='left')
user_merchant2 = pd.merge(user_merchant2, t4, on=['user_id', 'merchant_id'], how='left')
user_merchant2.user_merchant_buy_use_coupon = user_merchant2.user_merchant_buy_use_coupon.replace(np.nan, 0)
user_merchant2.user_merchant_buy_common = user_merchant2.user_merchant_buy_common.replace(np.nan, 0)
user_merchant2['user_merchant_coupon_transfer_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype(
    'float') / user_merchant2.user_merchant_received.astype('float')
user_merchant2['user_merchant_coupon_buy_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype(
    'float') / user_merchant2.user_merchant_buy_total.astype('float')
user_merchant2['user_merchant_rate'] = user_merchant2.user_merchant_buy_total.astype(
    'float') / user_merchant2.user_merchant_any.astype('float')
user_merchant2['user_merchant_common_buy_rate'] = user_merchant2.user_merchant_buy_common.astype(
    'float') / user_merchant2.user_merchant_buy_total.astype('float')
user_merchant2.to_csv('data/user_merchant2.csv', index=None)

# for dataset2
all_user_merchant = feature1[['user_id', 'merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature1[['user_id', 'merchant_id', 'date']]
t = t[t.date != 'null'][['user_id', 'merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature1[['user_id', 'merchant_id', 'coupon_id']]
t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature1[['user_id', 'merchant_id', 'date', 'date_received']]
t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][['user_id', 'merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature1[['user_id', 'merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature1[['user_id', 'merchant_id', 'date', 'coupon_id']]
t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][['user_id', 'merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant1 = pd.merge(all_user_merchant, t, on=['user_id', 'merchant_id'], how='left')
user_merchant1 = pd.merge(user_merchant1, t1, on=['user_id', 'merchant_id'], how='left')
user_merchant1 = pd.merge(user_merchant1, t2, on=['user_id', 'merchant_id'], how='left')
user_merchant1 = pd.merge(user_merchant1, t3, on=['user_id', 'merchant_id'], how='left')
user_merchant1 = pd.merge(user_merchant1, t4, on=['user_id', 'merchant_id'], how='left')
user_merchant1.user_merchant_buy_use_coupon = user_merchant1.user_merchant_buy_use_coupon.replace(np.nan, 0)
user_merchant1.user_merchant_buy_common = user_merchant1.user_merchant_buy_common.replace(np.nan, 0)
user_merchant1['user_merchant_coupon_transfer_rate'] = user_merchant1.user_merchant_buy_use_coupon.astype(
    'float') / user_merchant1.user_merchant_received.astype('float')
user_merchant1['user_merchant_coupon_buy_rate'] = user_merchant1.user_merchant_buy_use_coupon.astype(
    'float') / user_merchant1.user_merchant_buy_total.astype('float')
user_merchant1['user_merchant_rate'] = user_merchant1.user_merchant_buy_total.astype(
    'float') / user_merchant1.user_merchant_any.astype('float')
user_merchant1['user_merchant_common_buy_rate'] = user_merchant1.user_merchant_buy_common.astype(
    'float') / user_merchant1.user_merchant_buy_total.astype('float')
user_merchant1.to_csv('data/user_merchant1.csv', index=None)


##################  generate training and testing set ################
def get_label(s):
    s = s.split(':')
    if s[0] == 'null':
        return 0
    elif (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]),
                                                                      int(s[1][6:8]))).days <= 15:
        return 1
    else:
        return -1


merchant3 = pd.read_csv('data/merchant3_feature.csv')
user3 = pd.read_csv('data/user3_feature.csv')
user_merchant3 = pd.read_csv('data/user_merchant3.csv')
dataset3 = pd.merge(merchant3, user3, on='user_id', how='left')
dataset3 = pd.merge(dataset3, user_merchant3, on=['user_id', 'merchant_id'], how='left')
dataset3.drop_duplicates(inplace=True)
print(dataset3.shape)

dataset3.user_merchant_buy_total = dataset3.user_merchant_buy_total.replace(np.nan, 0)
dataset3.user_merchant_any = dataset3.user_merchant_any.replace(np.nan, 0)
dataset3.user_merchant_received = dataset3.user_merchant_received.replace(np.nan, 0)
dataset3['is_weekend'] = dataset3.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
weekday_dummies = pd.get_dummies(dataset3.day_of_week)
weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
dataset3 = pd.concat([dataset3, weekday_dummies], axis=1)
dataset3.drop(['merchant_id', 'day_of_week', 'coupon_count'], axis=1, inplace=True)
dataset3 = dataset3.replace('null', np.nan)
dataset3.to_csv('data/dataset3.csv', index=None)

merchant2 = pd.read_csv('data/merchant2_feature.csv')
user2 = pd.read_csv('data/user2_feature.csv')
user_merchant2 = pd.read_csv('data/user_merchant2.csv')
dataset2 = pd.merge(merchant2, user2, on='user_id', how='left')
dataset2 = pd.merge(dataset2, user_merchant2, on=['user_id', 'merchant_id'], how='left')
dataset2.drop_duplicates(inplace=True)
print(dataset2.shape)

dataset2.user_merchant_buy_total = dataset2.user_merchant_buy_total.replace(np.nan, 0)
dataset2.user_merchant_any = dataset2.user_merchant_any.replace(np.nan, 0)
dataset2.user_merchant_received = dataset2.user_merchant_received.replace(np.nan, 0)
dataset2['is_weekend'] = dataset2.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
weekday_dummies = pd.get_dummies(dataset2.day_of_week)
weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
dataset2 = pd.concat([dataset2, weekday_dummies], axis=1)
dataset2['label'] = dataset2.date.astype('str') + ':' + dataset2.date_received.astype('str')
dataset2.label = dataset2.label.apply(get_label)
dataset2.drop(['merchant_id', 'day_of_week', 'date', 'date_received', 'coupon_id', 'coupon_count'], axis=1,
              inplace=True)
dataset2 = dataset2.replace('null', np.nan)
dataset2.to_csv('data/dataset2.csv', index=None)

merchant1 = pd.read_csv('data/merchant1_feature.csv')
user1 = pd.read_csv('data/user1_feature.csv')
user_merchant1 = pd.read_csv('data/user_merchant1.csv')
dataset1 = pd.merge(merchant1, user1, on='user_id', how='left')
dataset1 = pd.merge(dataset1, user_merchant1, on=['user_id', 'merchant_id'], how='left')
dataset1.drop_duplicates(inplace=True)
print(dataset1.shape)

dataset1.user_merchant_buy_total = dataset1.user_merchant_buy_total.replace(np.nan, 0)
dataset1.user_merchant_any = dataset1.user_merchant_any.replace(np.nan, 0)
dataset1.user_merchant_received = dataset1.user_merchant_received.replace(np.nan, 0)
dataset1['is_weekend'] = dataset1.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
weekday_dummies = pd.get_dummies(dataset1.day_of_week)
weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
dataset1 = pd.concat([dataset1, weekday_dummies], axis=1)
dataset1['label'] = dataset1.date.astype('str') + ':' + dataset1.date_received.astype('str')
dataset1.label = dataset1.label.apply(get_label)
dataset1.drop(['merchant_id', 'day_of_week', 'date', 'date_received', 'coupon_id', 'coupon_count'], axis=1,
              inplace=True)
dataset1 = dataset1.replace('null', np.nan)
dataset1.to_csv('data/dataset1.csv', index=None)
