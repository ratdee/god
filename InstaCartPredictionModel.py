# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:30:45 2019

@author: Admin
"""

import pandas as pd
import numpy as np


# reading data



market_prior_orders = pd.read_csv('order_products_prior.csv', dtype={
        'order_id': np.int32,
        'product_id': np.int32,
        'add_to_cart_order': np.int16,
        'reordered': np.int8})



print('Loaded prior orders')

market_prior_orders=market_prior_orders.loc[market_prior_orders['order_id']<=110720]

print('Loading orders')
market_orders = pd.read_csv( 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

market_orders=market_orders.loc[market_orders['order_id']<=110720]

print('Loading aisles info')
aisles = pd.read_csv('products.csv', engine='c',
                           usecols = ['product_id','aisle_id'],
                       dtype={'product_id': np.int32, 'aisle_id': np.int32})

pd.set_option('display.float_format', lambda x: '%.3f' % x)

print("\n Checking the loaded CSVs")
print("Prior orders:", market_prior_orders.shape)
print("Orders", market_orders.shape)
print("Aisles:", aisles.shape)


test  = market_orders[market_orders['eval_set'] == 'test' ]
user_ids = test['user_id'].values
market_orders = market_orders[market_orders['user_id'].isin(user_ids)]

print(test.shape)

# Calculate the Prior : p(reordered|product_id)

prior = pd.DataFrame(market_prior_orders.groupby('product_id')['reordered']     \
                     .agg([('number_of_orders',len),('sum_of_reorders','sum')]))

prior['prior_p'] = (prior['sum_of_reorders']+1)/(prior['number_of_orders']+2) # Informed Prior


print('Here is The Prior: our first guess of how probable it is that a product be reordered once it has been ordered.')
print(prior.head())

# merge everything into one dataframe and save any memory space

combined_features = pd.DataFrame()
combined_features = pd.merge(market_prior_orders, market_orders, on='order_id', how='right')

# slim down comb - 
combined_features.drop(['eval_set','order_dow','order_hour_of_day'], axis=1, inplace=True)
del market_prior_orders
del market_orders

combined_features = pd.merge(combined_features, aisles, on ='product_id', how = 'left')
del aisles

prior.reset_index(inplace = True)
combined_features = pd.merge(combined_features, prior, on ='product_id', how = 'left')
del prior

print('combined data in DataFrame comb')
print(combined_features.head())


recount = pd.DataFrame()
recount['reorder_c'] = combined_features.groupby(combined_features.order_id)['reordered'].sum().fillna(0)
bins = [-0.1, 0, 2,4,6,8,11,14,19,71]
cat =  ['None','<=2','<=4','<=6','<=8','<=11','<=14','<=19','>19']
recount['reorder_b'] = pd.cut(recount['reorder_c'], bins, labels = cat)
recount.reset_index(inplace = True)

combined_features = pd.merge(combined_features, recount, how = 'left', on = 'order_id')
del recount
print(combined_features.head(50))

bins = [0,2,3,5,7,9,12,17,80]
cat = ['<=2','<=3','<=5','<=7','<=9','<=12','<=17','>17']

combined_features['atco1'] = pd.cut(combined_features['add_to_cart_order'], bins, labels = cat)
del combined_features['add_to_cart_order']
print('combined_features')
print(combined_features.head(50))

atco_fac = pd.DataFrame()
atco_fac = combined_features.groupby(['reordered', 'atco1'])['atco1'].agg(np.count_nonzero).unstack('atco1')

tot = pd.DataFrame()
tot = np.sum(atco_fac,axis=1)

atco_fac = atco_fac.iloc[:,:].div(tot, axis=0)
atco_fac = atco_fac.stack('atco1')
atco_fac = pd.DataFrame(atco_fac)
atco_fac.reset_index(inplace = True)
atco_fac.rename(columns = {0:'atco_fac_p'}, inplace = True)

combined_features = pd.merge(combined_features, atco_fac, how='left', on=('reordered', 'atco1'))
print(combined_features.head(50))


aisle_fac = pd.DataFrame()
aisle_fac = combined_features.groupby(['reordered', 'atco1', 'aisle_id'])['aisle_id']\
                .agg(np.count_nonzero).unstack('aisle_id')

tot = np.sum(aisle_fac,axis=1)

aisle_fac = aisle_fac.iloc[:,:].div(tot, axis=0)
aisle_fac = aisle_fac.stack('aisle_id')
aisle_fac = pd.DataFrame(aisle_fac)
aisle_fac.reset_index(inplace = True)
aisle_fac.rename(columns = {0:'aisle_fac_p'}, inplace = True)

combined_features = pd.merge(combined_features, aisle_fac, how = 'left', on = ('aisle_id','reordered','atco1'))
print(combined_features.head(50))


recount_fac = pd.DataFrame()
recount_fac = combined_features.groupby(['reordered', 'atco1', 'reorder_b'])['reorder_b']\
                    .agg(np.count_nonzero).unstack('reorder_b')

tot = pd.DataFrame()
tot = np.sum(recount_fac,axis=1)

recount_fac = recount_fac.iloc[:,:].div(tot, axis=0)
recount_fac.stack('reorder_b')
recount_fac = pd.DataFrame(recount_fac.unstack('reordered').unstack('atco1')).reset_index()
recount_fac.rename(columns = {0:'recount_fac_p'}, inplace = True)

combined_features = pd.merge(combined_features, recount_fac, how = 'left', on = ('reorder_b', 'reordered', 'atco1'))
print(recount_fac.head(50))


p = pd.DataFrame()
p = (combined_features.loc[:,'atco_fac_p'] * combined_features.loc[:,'aisle_fac_p'] * combined_features.loc[:,'recount_fac_p'])
p.reset_index()
combined_features['p'] = p

combined_features.head(30)

combined_features.to_csv('combined_features.csv', index=False)


# Calculate bf0 for products when first purchased aka reordered=0
comb0 = pd.DataFrame()
comb0 = combined_features[combined_features['reordered']==0]
comb0.loc[:,'first_order'] = comb0['order_number']
# now every product that was ordered has a posterior in usr.
comb0.loc[:,'beta'] = 1
comb0.loc[:,'bf'] = (comb0.loc[:,'prior_p'] * comb0.loc[:,'p']/(1 - comb0.loc[:,'p'])) # bf1
# Small 'slight of hand' here. comb0.bf is really the first posterior and second prior.

# Calculate beta and BF1 for the reordered products
comb1 = pd.DataFrame()
comb1 = combined_features[combined_features['reordered']==1]

comb1.loc[:,'beta'] = (1 - .05*comb1.loc[:,'days_since_prior_order']/30)
comb1.loc[:,'bf'] = (1 - comb1.loc[:,'p'])/comb1.loc[:,'p'] # bf0


comb_last = pd.DataFrame()
comb_last = pd.concat([comb0, comb1], axis=0).reset_index(drop=True)

comb_last = comb_last[['reordered', 'user_id', 'order_id', 'product_id','reorder_c','order_number',
                       'bf','beta','atco_fac_p', 'aisle_fac_p', 'recount_fac_p']]
comb_last = comb_last.sort_values((['user_id', 'order_number', 'bf']))

pd.set_option('display.float_format', lambda x: '%.6f' % x)
comb_last.head()


first_order = pd.DataFrame()
first_order = comb_last[comb_last.reordered == 0]
first_order.rename(columns = {'order_number':'first_o'}, inplace = True)
first_order.loc[:,'last_o'] = comb_last.groupby(['user_id'])['order_number'].transform(max)
first_order = first_order[['user_id','product_id','first_o','last_o']]

comb_last = pd.merge(comb_last, first_order, on = ('user_id', 'product_id'), how = 'left')
comb_last.head()




comb_last.to_csv('comb_last.csv', index=False)


temp = pd.pivot_table(comb_last[(comb_last.user_id == 3) & (comb_last.first_o == comb_last.order_number)],
                     values = 'bf', index = ['user_id', 'product_id'],
                     columns = 'order_number', dropna=False)
temp.head(10)

temp = temp.fillna(method='pad', axis=1).fillna(1)
temp.head(10)


pd.pivot_table(comb_last[comb_last.first_o <= comb_last.order_number],
                              values = 'bf', index = ['user_id', 'product_id'],
                              columns = 'order_number').head(10)


temp.update(pd.pivot_table(comb_last[comb_last.first_o <= comb_last.order_number],
                              values = 'bf', index = ['user_id', 'product_id'],
                              columns = 'order_number'))
temp.head(10)




print("Started Posterior calculations")

pred = pd.DataFrame(columns=['user_id', 'product_id'])
# comb_last_temp = pd.DataFrame()
# com = pd.DataFrame()

for uid in comb_last.user_id.unique():
    if uid % 1000 == 0:
        print("Posterior calculated until user %d" % uid)
      
    
#     del comb_last_temp
    comb_last_temp = pd.DataFrame()
    comb_last_temp = comb_last[comb_last['user_id'] == uid].reset_index()
    
#     del com
    com = pd.DataFrame()
    com = pd.pivot_table(comb_last_temp[comb_last_temp.first_o == comb_last_temp.order_number],
                         values = 'bf', index = ['user_id', 'product_id'],
                         columns = 'order_number', dropna=False)
    com = com.fillna(method='pad', axis=1).fillna(1)
    com.update(pd.pivot_table(comb_last_temp[comb_last_temp.first_o <= comb_last_temp.order_number],
                              values = 'bf', index = ['user_id', 'product_id'],
                              columns = 'order_number'))
com.reset_index(inplace=True)
com['posterior'] = com.product(axis=1)
    
pred = pred.append(com.sort_values(by=['posterior'], ascending=False).head(10)    \
                       .groupby('user_id')['product_id'].apply(list).reset_index())    

print("Posterior calculated for all users")

pred = pred.rename(columns={'product_id': 'products'})
#print(pred.head())

print('before writing to csv')

pred.to_csv('InstaMarketPredictions.csv', index=False)







