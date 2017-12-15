"""
 This is for the 2017-12-12 data clean task.
 Today I set the threshold, to generate the result based on the rule:
 1.Calculate the click/addToCart/buy/addFavorite counts
 2.Set threshold and get the total possibility of the repeat buyer
 3.Generate the version 1 result, get 0.58 score and ranked 63 in Tianchi

 -- Haibo Yu
"""
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

user_log_data = pd.DataFrame(pd.read_csv('./data/user_log_format1.csv'))
user_log_data['clickCounts'] = (user_log_data.action_type == 0).astype(int)
user_log_data['addToCartCounts'] = (user_log_data.action_type == 1).astype(int)
user_log_data['buyCounts'] = (user_log_data.action_type == 2).astype(int)
user_log_data['addFavCounts'] = (user_log_data.action_type == 3).astype(int)

groupSum = user_log_data.groupby(['user_id','seller_id']).agg({'clickCounts':'sum', 'addToCartCounts':'sum','buyCounts':'sum', 'addFavCounts':'sum'})
tempResult = groupSum.reset_index()

tempResult['prob'] = tempResult['clickCounts']*0.05 + tempResult['addToCartCounts']*0.1 + tempResult['buyCounts']*0.5 + tempResult['addFavCounts']*0.2

tempResult.loc[tempResult['prob'] > 1, 'prob'] = 1
del tempResult['clickCounts']
del tempResult['addToCartCounts']
del tempResult['buyCounts']
del tempResult['addFavCounts']
result = tempResult[tempResult['prob']>0.8]
result.drop_duplicates()
result.round(2)

result.to_csv("result.csv",index=False, float_format='%.2f')
