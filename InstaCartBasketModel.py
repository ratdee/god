# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:04:58 2019

@author: Admin
"""

# -*- coding: utf-8 -*-

"""
Created on Mon July 8 17:30:45 2019

@author: Admin
"""

import pandas as pd
import numpy as np


# reading data




order_products_prior_df = pd.read_csv('order_products_prior.csv', dtype={
        'order_id': np.int32,
        'product_id': np.int32,
        'add_to_cart_order': np.int16,
        'reordered': np.int8})



print('Loaded prior orders')

print('shape of Ordersproduct priors',order_products_prior_df.shape)

order_products_prior_df=order_products_prior_df.loc[order_products_prior_df['order_id']<=2110720]

print('Loading orders')
orders_df = pd.read_csv( 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})




orders_df=orders_df.loc[orders_df['order_id']<=2110720]

print(orders_df.shape)

print('Loading aisles info')

aisles = pd.read_csv('products.csv', engine='c',
                           usecols = ['product_id','aisle_id'],
                       dtype={'product_id': np.int32, 'aisle_id': np.int32})

pd.set_option('display.float_format', lambda x: '%.3f' % x)

print("\n Checking the loaded CSVs")
print("Prior orders:", order_products_prior_df.shape)
print("Orders", orders_df.shape)
print("Aisles:", aisles.shape)


test  = orders_df[orders_df['eval_set'] == 'test' ]
user_ids = test['user_id'].values
orders_df = orders_df[orders_df['user_id'].isin(user_ids)]

print('test shape', test.shape)

print(orders_df.shape)


prior = pd.DataFrame(order_products_prior_df.groupby('product_id')['reordered']     \
                     .agg([('number_of_orders',len),('sum_of_reorders','sum')]))

print(prior.head())

prior['prior_p'] = (prior['sum_of_reorders']+1)/(prior['number_of_orders']+2) # Informed Prior

print(prior.head())

print('Here is The Prior: our first guess of how probable it is that a product be reordered once it has been ordered.')
#print(prior.head())

# merge everything into one dataframe and save any memory space

combined_features = pd.DataFrame()
combined_features = pd.merge(order_products_prior_df, orders_df, on='order_id', how='right')

# slim down comb - 
combined_features.drop(['eval_set','order_dow','order_hour_of_day'], axis=1, inplace=True)
del order_products_prior_df
del orders_df

combined_features = pd.merge(combined_features, aisles, on ='product_id', how = 'left')
del aisles

prior.reset_index(inplace = True)
combined_features = pd.merge(combined_features, prior, on ='product_id', how = 'left')
del prior


#print(combined_features.head())


recount = pd.DataFrame()
recount['reorder_c'] = combined_features.groupby(combined_features.order_id)['reordered'].sum().fillna(0)

#print(recount.head(20))

print('classification')
bins = [-0.1, 0, 2,4,6,8,11,14,19,71]
cat =  ['None','<=2','<=4','<=6','<=8','<=11','<=14','<=19','>19']
recount['reorder_b'] = pd.cut(recount['reorder_c'], bins, labels = cat)
recount.reset_index(inplace = True)

#print(recount.head(20))



#We discretize reorder count into categories, 9 buckets, being sure to include 0 as bucket. These bins maximize mutual information with ['reordered'].
combined_features = pd.merge(combined_features, recount, how = 'left', on = 'order_id')
del recount
#print(combined_features.head(50))


bins = [0,2,3,5,7,9,12,17,80]
cat = ['<=2','<=3','<=5','<=7','<=9','<=12','<=17','>17']

combined_features['atco1'] = pd.cut(combined_features['add_to_cart_order'], bins, labels = cat)
del combined_features['add_to_cart_order']

#print(combined_features.head(50))

combined_features.to_csv('combined_features.csv', index=False)

atco_fac = pd.DataFrame()
atco_fac = combined_features.groupby(['reordered', 'atco1'])['atco1'].agg(np.count_nonzero).unstack('atco1')

#print(atco_fac.head(10))


tot = np.sum(atco_fac,axis=1)
print(tot.head(10))

atco_fac = atco_fac.iloc[:,:].div(tot, axis=0)
#print(atco_fac.head(10))



atco_fac = atco_fac.stack('atco1')


#print(atco_fac.head(20))

atco_fac = pd.DataFrame(atco_fac)
atco_fac.reset_index(inplace = True)
atco_fac.rename(columns = {0:'atco_fac_p'}, inplace = True)

combined_features = pd.merge(combined_features, atco_fac, how='left', on=('reordered', 'atco1'))
combined_features.head(50)


aisle_fac = pd.DataFrame()
aisle_fac = combined_features.groupby(['reordered', 'atco1', 'aisle_id'])['aisle_id']\
                .agg(np.count_nonzero).unstack('aisle_id')
              
print(aisle_fac.head(30))



#print(aisle_fac.head(30))
tot = np.sum(aisle_fac,axis=1)

print(tot.head(20))

aisle_fac = aisle_fac.iloc[:,:].div(tot, axis=0)

print(aisle_fac.head(20))
print('Stacking  Aisle Fac')
aisle_fac = aisle_fac.stack('aisle_id')
print(aisle_fac.head(20))
aisle_fac = pd.DataFrame(aisle_fac)
aisle_fac.reset_index(inplace = True)
aisle_fac.rename(columns = {0:'aisle_fac_p'}, inplace = True)

combined_features = pd.merge(combined_features, aisle_fac, how = 'left', on = ('aisle_id','reordered','atco1'))

recount_fac = pd.DataFrame()
recount_fac = combined_features.groupby(['reordered', 'atco1', 'reorder_b'])['reorder_b']\
                    .agg(np.count_nonzero).unstack('reorder_b')
print(recount_fac.head(20))                   
                    
tot = pd.DataFrame()
tot = np.sum(recount_fac,axis=1)  

print(tot.head(20))    
recount_fac = recount_fac.iloc[:,:].div(tot, axis=0)

print(recount_fac.head(20))  


#print('after stacking***************************')
recount_fac.stack('reorder_b')
print(recount_fac.head(20))  
recount_fac = pd.DataFrame(recount_fac.unstack('reordered').unstack('atco1')).reset_index()
#print(recount_fac.head(20)) 


recount_fac.rename(columns = {0:'recount_fac_p'}, inplace = True)

combined_features = pd.merge(combined_features, recount_fac, how = 'left', on = ('reorder_b', 'reordered', 'atco1'))
print(recount_fac.head(50))
print(combined_features.head(20))




p = pd.DataFrame()
p = (combined_features.loc[:,'atco_fac_p'] * combined_features.loc[:,'aisle_fac_p'] * combined_features.loc[:,'recount_fac_p'])
p.reset_index()


combined_features['p'] = p

print(combined_features.head(30))

comb0 = pd.DataFrame()         

print(combined_features.shape)
comb0 = combined_features[combined_features['reordered']==0]
print(comb0.shape)
comb0.loc[:,'first_order'] = comb0['order_number']
# now every product that was ordered has a posterior in usr.
comb0.loc[:,'beta'] = 1
comb0.loc[:,'bf'] = (comb0.loc[:,'prior_p'] * comb0.loc[:,'p']/(1 - comb0.loc[:,'p'])) # bf1
# Small 'slight of hand' here. comb0.bf is really the first posterior and second prior.


#comb0.to_csv('comb0.csv', index=False)

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

first_order.to_csv('first_order_before_transform.csv', index=False)

first_order.loc[:,'last_o'] = comb_last.groupby(['user_id'])['order_number'].transform(max)

first_order.to_csv('first_order_transform.csv', index=False)
first_order = first_order[['user_id','product_id','first_o','last_o']]

comb_last = pd.merge(comb_last, first_order, on = ('user_id', 'product_id'), how = 'left')
comb_last.head()

comb_last.to_csv('comb_last.csv')
comb_last = pd.read_csv('comb_last.csv', index_col=0)


#comb_last.to_csv('comb_last.csv', index=False)
temp = pd.pivot_table(comb_last[(comb_last.user_id == 786
) & (comb_last.first_o == comb_last.order_number)],
                     values = 'bf', index = ['user_id', 'product_id'],
                     columns = 'order_number', dropna=False)
#print (temp.head(10))

temp = temp.fillna(method='pad', axis=1).fillna(1)
temp.head(10)

temp.to_csv('temp.csv')


#print(pd.pivot_table(comb_last[comb_last.first_o <= comb_last.order_number],
#                              values = 'bf', index = ['user_id', 'product_id'],
 #                             columns = 'order_number').head(10))


temp.update(pd.pivot_table(comb_last[comb_last.first_o <= comb_last.order_number],
                              values = 'bf', index = ['user_id', 'product_id'],
                              columns = 'order_number'))


print(temp.head(10))
#temp.to_csv('temp.csv')


import logging
logging.basicConfig(filename='bayes.log',level=logging.DEBUG)
logging.debug("Started Posterior calculations")
print("Started Posterior calculations")
pred = pd.DataFrame(columns=['user_id', 'product_id'])
pred['user_id'] = pred.user_id.astype(np.int32)
pred['product_id'] = pred.product_id.astype(np.int32)






for uid in comb_last.user_id.unique():
    if uid % 1000 == 0:
        print("Posterior calculated until user %d" % uid)
        logging.debug("Posterior calculated until user %d" % uid)
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
logging.debug("Posterior calculated for all users")
pred = pred.rename(columns={'product_id': 'products'})
print(pred.head())
pred.to_csv('Finalpredictions.csv', index=False)
         
pred = pred.merge(test, on='user_id', how='left')[['order_id', 'products']]
pred['products'] = pred['products'].apply(lambda x: [int(i) for i in x])    \
                    .astype(str).apply(lambda x: x.strip('[]').replace(',', ''))
print(pred.head())

pred.to_csv('Testpredictions.csv', index=False)