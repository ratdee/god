# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:37:10 2019

@author:Imarticus Machine Learning Team
"""

import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()


pd.options.mode.chained_assignment = None  # default='warn'


order_products_test_df=pd.read_csv("order_products_test.csv")
order_products_train_df=pd.read_csv("order_products_train.csv")
order_products_train_df=order_products_train_df.loc[order_products_train_df['order_id']<=2110720]
order_products_prior_df = pd.read_csv("order_products_prior.csv")
order_products_prior_df=order_products_prior_df.loc[order_products_prior_df['order_id']<=2110720]

orders_df = pd.read_csv("orders.csv")

orders_df=orders_df.loc[orders_df['order_id']<=110720]
products_df = pd.read_csv("products.csv")
aisles_df = pd.read_csv("aisles.csv")
departments_df = pd.read_csv("departments.csv")

#print(orders_df.head())

#print(order_products_prior_df.head())




#Problem Set 1a)	When do customers order the most?
#Day of the week


plt.figure(figsize=(12,8))
sns.countplot(x="order_dow", data=orders_df, color=color[0])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of order by week day", fontsize=15)
plt.show()


df_top_freq_dow = orders_df.groupby(['order_dow'])['order_id'].agg({"code_count": len}).sort_values("code_count", ascending=False).head().reset_index()
#print(df_top_freq)
print('frequency of order of Week')
print(df_top_freq_dow.head())
#Both the plot and groupby  query confirm that date of week 0 has the most orders
#   order_dow  code_count
#0          0       12342
#1          1       12214
#2          2        9828
#3          5        9332
#4          6        9251


#Problem Set 1b) Time of the day in Seaborne Plot
plt.figure(figsize=(12,8))
sns.countplot(x="order_hour_of_day", data=orders_df, color=color[1])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Hour of day', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of order by hour of day", fontsize=15)
plt.show()


df_top_freq_ohd = orders_df.groupby(['order_hour_of_day'])['order_id'].agg({"code_count": len}).sort_values("code_count", ascending=False).head().reset_index()
#print(df_top_freq)
print('Time of the day')
print(df_top_freq_ohd.head())

#Problem Set 1b)Time of the day results by query
 #  order_hour_of_day  code_count
#0                 10        5968
#1                 15        5859
#2                 14        5858
#3                 11        5853
#4                 13        5775
 
#Both the plot and groupby  query confirm that hour of day  10 has the most orders
 
#Problem Set 1c)	Combination of the Day of the Week and Time of the day
grouped_df = orders_df.groupby(["order_dow", "order_hour_of_day"])["order_id"].aggregate("count").reset_index()
grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_id')

plt.figure(figsize=(12,6))
sns.heatmap(grouped_df)
plt.title("Frequency of Day of week Vs Hour of day")
plt.show()


df_top_freq_dow_ohd = orders_df.groupby(['order_dow', 'order_hour_of_day'])['order_id'].agg({"code_count": len}).sort_values("code_count", ascending=False).head().reset_index()
#print(df_top_freq)
print('Order hour of day and Time of the day')
print(df_top_freq_dow_ohd.head())



order_products_prior_df = pd.merge(order_products_prior_df, products_df, on='product_id', how='left')
order_products_train_df = pd.merge(order_products_train_df, products_df, on='product_id', how='left')
order_products_test_df = pd.merge(order_products_test_df, products_df, on='product_id', how='left')



del products_df
order_products_prior_df = pd.merge(order_products_prior_df, aisles_df, on='aisle_id', how='left')
order_products_train_df = pd.merge(order_products_train_df, aisles_df, on='aisle_id', how='left')
order_products_test_df = pd.merge(order_products_test_df, aisles_df, on='aisle_id', how='left')


del aisles_df


order_products_prior_df = pd.merge(order_products_prior_df, departments_df, on='department_id', how='left')
order_products_train_df = pd.merge(order_products_train_df, departments_df, on='department_id', how='left')
order_products_test_df = pd.merge(order_products_test_df, departments_df, on='department_id', how='left')
del departments_df
print(order_products_prior_df.head())

#Problem statement 2
#Top5 products in Prior Data Set
print('Top 5 products in Prior DataSet')
cnt_srs = order_products_prior_df['product_name'].value_counts().reset_index().head(5)
cnt_srs.columns = ['product_name', 'frequency_count']
print(cnt_srs)


reorder_products_prior_df=order_products_prior_df.loc[order_products_prior_df['reordered']==1]


print('Top5 Reorder Products in prior dataset')
reorder_cnt_srs = reorder_products_prior_df['product_name'].value_counts().reset_index().head(5)
reorder_cnt_srs.columns = ['product_name', 'frequency_count']
print(reorder_cnt_srs)



plt.figure(figsize=(10,10))
temp_series = order_products_prior_df['department'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
plt.pie(sizes, labels=labels, 
        autopct='%1.1f%%', startangle=200)
plt.title("Departments distribution", fontsize=15)
plt.show()


#Problem Set 3)	What is the reorder ratio for each department in Prior Data Set
grouped_df = order_products_prior_df.groupby(["department"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['department'].values, grouped_df['reordered'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.title("Department wise reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


print(grouped_df.head(30))




#Problem Set 2 and 3 from Training DataSet

print('Top 5 products from Training Data Set')
cnt_srs = order_products_train_df['product_name'].value_counts().reset_index().head(5)
cnt_srs.columns = ['product_name', 'frequency_count']
print(cnt_srs)



#Top 5 reordered products in prior DataSet


reorder_products_train_df=order_products_train_df.loc[order_products_train_df['reordered']==1]


print('Top5 Reorder products from Training DataSet')
reorder_cnt_srs = reorder_products_train_df['product_name'].value_counts().reset_index().head(5)
reorder_cnt_srs.columns = ['product_name', 'frequency_count']
print(reorder_cnt_srs)

#Reorder ratio in department in training
grouped_df = order_products_train_df.groupby(["department"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['department'].values, grouped_df['reordered'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.title("Department wise reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

print(grouped_df.head(30))




#Testing DataSets

print('Top 5 products from Testing')
cnt_srs = order_products_test_df['product_name'].value_counts().reset_index().head(5)
cnt_srs.columns = ['product_name', 'frequency_count']
print(cnt_srs)




#Top 5 reordered products from  Testing DataSet


reorder_products_test_df=order_products_test_df.loc[order_products_test_df['reordered']==1]


print('Top5 reorder products from Testing')
reorder_cnt_srs = reorder_products_test_df['product_name'].value_counts().reset_index().head(5)
reorder_cnt_srs.columns = ['product_name', 'frequency_count']
print(reorder_cnt_srs)



#Reorder ratio of  department in Testing DataSet
grouped_df = order_products_test_df.groupby(["department"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['department'].values, grouped_df['reordered'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.title("Department wise reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

print(grouped_df.head(30))

 