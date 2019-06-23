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
order_products_train_df=order_products_train_df.loc[order_products_train_df['order_id']<=110720]
order_products_prior_df = pd.read_csv("order_products_prior.csv")
order_products_prior_df=order_products_prior_df.loc[order_products_prior_df['order_id']<=110720]

orders_df = pd.read_csv("orders.csv")

orders_df=orders_df.loc[orders_df['order_id']<=110720]
products_df = pd.read_csv("products.csv")
aisles_df = pd.read_csv("aisles.csv")
departments_df = pd.read_csv("departments.csv")

#print(orders_df.head())

#print(order_products_prior_df.head())

cnt_srs = orders_df.eval_set.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Eval set type', fontsize=12)
plt.title('Count of rows in each dataset', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

def get_unique_count(x):
    return len(np.unique(x))

cnt_srs = orders_df.groupby("eval_set")["user_id"].aggregate(get_unique_count)
print(cnt_srs)


cnt_srs = orders_df.groupby("eval_set")["user_id"].aggregate(get_unique_count)
print(cnt_srs)


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


#Order hour of day and Time of the day
 #  order_dow  order_hour_of_day  code_count
#0          0                 14        1156
#1          1                 10        1151
#2          0                 15        1115
#3          0                 13        1105
#4          1                  9        1070

#Both the plot and groupby  query confirm that Day/hour is 0 and 14



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
cnt_srs = order_products_prior_df['product_name'].value_counts().reset_index().head(20)
cnt_srs.columns = ['product_name', 'frequency_count']
print(cnt_srs)



#0                     Banana            15450
#1     Bag of Organic Bananas            12410
#2       Organic Strawberries             8528
#3       Organic Baby Spinach             7852
#4       Organic Hass Avocado             6857


#Top 5 reordered products in Prior Data Set

reorder_products_prior_df=order_products_prior_df.loc[order_products_prior_df['reordered']==1]


print('Top5 Reorder Products in prior dataset')
reorder_cnt_srs = reorder_products_prior_df['product_name'].value_counts().reset_index().head(20)
reorder_cnt_srs.columns = ['product_name', 'frequency_count']
print(reorder_cnt_srs)


#0                     Banana            13024
#1     Bag of Organic Bananas            10298
#2       Organic Strawberries             6603
#3       Organic Baby Spinach             6093
#4       Organic Hass Avocado             5435





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

#       department  reordered
#0           alcohol   0.568163
#1            babies   0.582287
#2            bakery   0.634430
#3         beverages   0.651032
#4         breakfast   0.559873
#5              bulk   0.581642
#6      canned goods   0.461429
#7        dairy eggs   0.669321
#8              deli   0.605009
#9   dry goods pasta   0.462253
#10           frozen   0.539004
#11        household   0.408212
#12    international   0.361472
#13     meat seafood   0.564596
#14          missing   0.397946
#15            other   0.434746
#16           pantry   0.347366
#17    personal care   0.323112
#18             pets   0.626437
#19          produce   0.651553
#20           snacks   0.575681


#Problem Set 2 and 3 from Training DataSet

print('Top 5 products from Training Data Set')
cnt_srs = order_products_train_df['product_name'].value_counts().reset_index().head(20)
cnt_srs.columns = ['product_name', 'frequency_count']
print(cnt_srs)


#0                   Banana              463
#1   Bag of Organic Bananas              387
#2     Organic Strawberries              292
#3     Organic Baby Spinach              232
#4              Large Lemon              198


#Top 5 reordered products in prior DataSet


reorder_products_train_df=order_products_train_df.loc[order_products_train_df['reordered']==1]


print('Top5 Reorder products from Training DataSet')
reorder_cnt_srs = reorder_products_train_df['product_name'].value_counts().reset_index().head(20)
reorder_cnt_srs.columns = ['product_name', 'frequency_count']
print(reorder_cnt_srs)

#Top5 Reorder products in Training
#0                                 Banana              409
#1                 Bag of Organic Bananas              340
#2                   Organic Strawberries              233
#3                   Organic Baby Spinach              188
#4                        Organic Avocado              162

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

#         department  reordered
#0           alcohol   0.609023
#1            babies   0.561772
#2            bakery   0.628237
#3         beverages   0.659265
#4         breakfast   0.586777
#5              bulk   0.441176
#6      canned goods   0.488215
#7        dairy eggs   0.670050
#8              deli   0.615036
#9   dry goods pasta   0.484536
#10           frozen   0.558496
#11        household   0.426637
#12    international   0.348148
#13     meat seafood   0.593039
#14          missing   0.351852
#15            other   0.351351
#16           pantry   0.346361
#17    personal care   0.303823
#18             pets   0.532609
#19          produce   0.663078
#20           snacks   0.568092



#Testing DataSets

print('Top 5 products from Testing')
cnt_srs = order_products_test_df['product_name'].value_counts().reset_index().head(20)
cnt_srs.columns = ['product_name', 'frequency_count']
print(cnt_srs)


#Top 5 products from Testing DataSet
 #             product_name  frequency_count
#0                   Banana             4217
#1   Bag of Organic Bananas             3541
#2     Organic Strawberries             2512
#3     Organic Baby Spinach             2198
#4              Large Lemon             1819


#Top 5 reordered products from  Testing DataSet


reorder_products_test_df=order_products_test_df.loc[order_products_test_df['reordered']==1]


print('Top5 reorder products from Testing')
reorder_cnt_srs = reorder_products_test_df['product_name'].value_counts().reset_index().head(20)
reorder_cnt_srs.columns = ['product_name', 'frequency_count']
print(reorder_cnt_srs)

#Top5 reorder products from Testing DataSet
 #                 product_name  frequency_count
#0                       Banana             3715
#1       Bag of Organic Bananas             3032
#2         Organic Strawberries             1981
#3         Organic Baby Spinach             1825
#4         Organic Hass Avocado             1436

#Reorder ratio in department in Testing DataSet
grouped_df = order_products_test_df.groupby(["department"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['department'].values, grouped_df['reordered'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.title("Department wise reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

print(grouped_df.head(30))

 #        department  reordered
#0           alcohol   0.610156
#1            babies   0.561956
#2            bakery   0.635846
#3         beverages   0.654762
#4         breakfast   0.585118
#5              bulk   0.600000
#6      canned goods   0.488751
#7        dairy eggs   0.674622
#8              deli   0.616821
#9   dry goods pasta   0.491148
#10           frozen   0.560110
#11        household   0.428485
#12    international   0.365280
#13     meat seafood   0.578657
#14          missing   0.366489
#15            other   0.372998
#16           pantry   0.358004
#17    personal care   0.337891
#18             pets   0.640408
#19          produce   0.663387
#20           snacks   0.585194