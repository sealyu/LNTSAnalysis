import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# user_log_data = pd.DataFrame(pd.read_csv('./data/user_log_format1.csv'))
user_log_data = pd.DataFrame(pd.read_csv('./data/user_log_format1.csv'))
# print(user_log_data.loc[user_log_data["gender"] == 1].size)
# grouped = user_log_data.groupby(['user_id','seller_id'])
# user_log_data.set_index(['user_id','seller_id'], inplace = True)

# print(user_log_data.info())

# r = user_log_data.groupby(['user_id','seller_id']).apply(lambda g: g.action_type.value_counts()).reset_index(name='action_count')
user_log_data['clickCounts'] = (user_log_data.action_type == 0).astype(int)
user_log_data['addToCartCounts'] = (user_log_data.action_type == 1).astype(int)
user_log_data['buyCounts'] = (user_log_data.action_type == 2).astype(int)
user_log_data['addFavCounts'] = (user_log_data.action_type == 3).astype(int)

groupSum = user_log_data.groupby(['user_id','seller_id']).agg({'clickCounts':'sum', 'addToCartCounts':'sum','buyCounts':'sum', 'addFavCounts':'sum'})
tempResult = groupSum.reset_index()

tempResult['prob'] = tempResult['clickCounts']*0.05 + tempResult['addToCartCounts']*0.1 + tempResult['buyCounts']*0.5 + tempResult['addFavCounts']*0.2
#toFloatValue = lambda x: x-1 if (x>1) else x
#result['prob'] = result['prob'].apply(toFloatValue)
tempResult.loc[tempResult['prob'] > 1, 'prob'] = 1
# result['prob'] = result['prob'].apply(lambda s : s-1 if (s>1) else s)）
# result[result['prob'] > 1]['prob'] = 1
# result['prob'] = result['prob'].astype('float64')
del tempResult['clickCounts']
del tempResult['addToCartCounts']
del tempResult['buyCounts']
del tempResult['addFavCounts']
result = tempResult[tempResult['prob']>0.8]
result.drop_duplicates()
result.round(2)
#print(result.head(50))
result.to_csv("result.csv",index=False, float_format='%.2f')
