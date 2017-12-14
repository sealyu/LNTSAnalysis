import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

user_log_data = pd.DataFrame(pd.read_csv('./data/user_log_format1.csv'))

user_log_data['clickCounts'] = (user_log_data.action_type == 0).astype(int)
user_log_data['addToCartCounts'] = (user_log_data.action_type == 1).astype(int)
user_log_data['buyCounts'] = (user_log_data.action_type == 2).astype(int)
user_log_data['addFavCounts'] = (user_log_data.action_type == 3).astype(int)

groupSum = user_log_data.groupby(['user_id','seller_id']).agg({'clickCounts':'sum', 'addToCartCounts':'sum','buyCounts':'sum', 'addFavCounts':'sum'})
tempResult = groupSum.reset_index()

tempResult = tempResult[tempResult['clickCounts'] < 100 | ((tempResult['clickCounts']>=100) & ((tempResult['addToCartCounts'] !=0) | (tempResult['buyCounts']!=0) | (tempResult['addFavCounts']!=0)))]

# print(filterClick.head(50))

tempResult['prob'] = tempResult['clickCounts']*0.01 + tempResult['addToCartCounts']*0.2 + tempResult['buyCounts']*0.5 + tempResult['addFavCounts']*0.4
tempResult.loc[tempResult['prob'] > 1, 'prob'] = 1

del tempResult['clickCounts']
del tempResult['addToCartCounts']
del tempResult['buyCounts']
del tempResult['addFavCounts']
result = tempResult[tempResult['prob']>=0.7]
result.drop_duplicates()
result.round(2)

result.to_csv("result.csv",index=False, float_format='%.2f')


