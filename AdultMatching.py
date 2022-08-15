#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import pipeline_dp as pdp
import math
import random
import data_tools_syr3 as syr
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics, tree 
from scipy import stats
import timeit
import xlwt
from xlwt import Workbook
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
wbboh=Workbook()
#prepare hist
def prep_hist(df):
  mar_df_1=df.groupby(['PUMA','AGE','RACE','SEX','OWNERSHP','INCTOT_DECILE'])[['GQ']]
  hist=mar_df_1.count()
  hist.reset_index()
  hist1=hist.copy()
  for i in range(6):
    hist1=hist1.reset_index(level=0)
  #print(hist1)
  hist1=hist1.rename(columns={'GQ':"COUNT"})
  return hist1
def prep_hist2(df,fnames):
  mar_df_1=df.groupby(fnames)[['INDNAICS']]
  hist=mar_df_1.count()
  hist.reset_index()
  hist1=hist.copy()
  for i in range(len(fnames)):
    hist1=hist1.reset_index(level=0)
  #print(hist1)
  hist1=hist1.rename(columns={'INDNAICS':"COUNT"})
  return hist1
def prep_hist3(df,fnames,cntname):
  mar_df_1=df.groupby(fnames)[[cntname]]
  hist=mar_df_1.count()
  hist.reset_index()
  hist1=hist.copy()
  for i in range(len(fnames)):
    hist1=hist1.reset_index(level=0)
  #print(hist1)
  hist1=hist1.rename(columns={cntname:"COUNT"})
  return hist1
def prep_hist_compas(df,fnames):
  mar_df_1=df.groupby(fnames)[['Case_ID']]
  hist=mar_df_1.count()
  hist.reset_index()
  hist1=hist.copy()
  for i in range(len(fnames)):
    hist1=hist1.reset_index(level=0)
  #print(hist1)
  hist1=hist1.rename(columns={'Case_ID':"COUNT"})
  return hist1
#create function to replicate rows based on GQ value
def count_scale_df(hist):
  hist_r=hist.copy()
  for index, row in hist.iterrows():
    #print(row['COUNT'])
    for i in range(int(row['COUNT'])):
      hist_r=hist_r.append(row)
  return hist_r
#
def count_scale_df2(hist):
  hist_r=hist.copy()
  concat_rows=[]
  for index, row in hist.iterrows():
    #print(row['COUNT'])
    for i in range(int(row['COUNT'])):
      concat_rows.append(row)
  hist_r=pd.concat(concat_rows)
  return hist_r

def redact_counts_df(hist,k): #Outputs redact-suppressed histogram
  hist1=hist.copy()
  counts_1=list(hist1['COUNT'])
  print("counts:",counts_1)
  #print("Hola:",counts_1)
  for i in range(len(counts_1)):
    if counts_1[i]<k:
      counts_1[i]=(k//2)
  hist1['COUNT']=counts_1
  return hist1
#
def dp_counts_df(hist,eps): #Outputs redact-suppressed histogram
  hist1=hist.copy()
  counts_1=list(hist1['COUNT'])
  #try:
    #dp_counts=add_dp_hist(counts_1,eps)
  #except:
  dp_counts=syr.add_dp_hist(counts_1,eps)
  hist1['COUNT']=dp_counts
  return hist1
#
def randmerge_counts_df(hist,k): #Outputs redact-suppressed histogram
  hist1=hist.copy()
  #start=timeit.default_timer()
  counts_1=list(hist1['COUNT'])
  #try:
  dp_counts=syr.spread_k_anonymise(counts_1,k)
  #except:
    #dp_counts=syr.spread_k_anonymise(counts_1,k)
  hist1['COUNT']=dp_counts
  #stop=timeit.default_timer()
  #print('Time: ',stop-start)
  return hist1

def adjmerge_counts_df(hist,k):
  hist1=hist.copy()
  #start=timeit.default_timer()
  counts_1=list(hist1['COUNT'])
  #try:
  dp_counts=syr.adj_k_anonymise(counts_1,k)
  #except:
    #dp_counts=syr.spread_k_anonymise(counts_1,k)
  hist1['COUNT']=dp_counts
  #stop=timeit.default_timer()
  #print('Time: ',stop-start)
  return hist1
#######
#SOLID! Compares optimal heights for each race over different methods
#df1=pd.read_csv("adult.csv")
#pumas=list(df1['PUMA'].unique())
#race='White'
sheet2=wbboh.add_sheet('Adult')
sheet2.write(0, 1, 'Race')
sheet2.write(0, 2, 'PUMA')
sheet2.write(0, 3, 'Unperturbed Height')
sheet2.write(0, 4, 'Redact 6')
sheet2.write(0, 5, 'Redact 10')
sheet2.write(0, 6, 'DP 1')
sheet2.write(0, 7, 'DP 2')
ct=0
for race in ['Black', 'White', 'Asian-Pac-Islander', 'Other','Amer-Indian-Eskimo']:
  ct+=1
  df=pd.read_csv('adult.csv')
  df = df[(df.astype(str) != ' ?').all(axis=1)]
  # Create a new income_bi column
  df['income_bi'] = df.apply(lambda row: 1 if '>50K'in row['income'] else 0, axis=1)
  # Remove redundant columns
  df = df.drop(['income','capital-gain','capital-loss','native-country'], axis=1)
  df=df[df['race']==race]
  df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender'])
  #sheet3=wbb.add_sheet('TX Feat Imp - Redact 10')
  #sheet2.write(0, 0, 'PUMAs')
  #pumas=[pumas[1]]
  #for puma in pumas:
  #for race in range(1):
  #  ct+=1
  #sheet2.write(ct,2,puma)
  print('\nRace - ',race)
  sheet2.write(ct,0,'Adult')
  sheet2.write(ct,1,race)
  #print('\nRACE = ',race)
  puma_1_df=df.copy()
  #puma_1_df=puma_1_df[(puma_1_df['race']==race)]#&(puma_1_df['PUMA']==puma)]
  #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
  feature_cols=list(puma_1_df.columns)
  #print(feature_cols)
  feature_cols.remove('fnlwgt')
  #puma_1_df=prep_hist(puma_1_df)
  puma_1_df=prep_hist3(puma_1_df,feature_cols,'fnlwgt')
  feature_cols.remove('income_bi')
  #print(puma_1_df['COUNT'])
  #puma_1_df=redact_counts_df(puma_1_df,k)
  puma_1_df=count_scale_df(puma_1_df)
  #feature_cols.remove('income')
  #feature_cols=['AGE','SEX','RACE','OWNERSHP']
  #print(puma_1_df.columns)
  X=puma_1_df[feature_cols]
  y=puma_1_df['income_bi']
  #print('\n',feature_cols)
  np.random.seed(0)
  #try:
  X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
  #except:
  #  sheet2.write(ct,2,'Beep')
    #print('Not enough individuals')
  #  continue
  #print(y_test)
  #np.random.seed(0)
  dt_1=DecisionTreeClassifier(random_state=0)#, max_depth=2)
  dt_1=dt_1.fit(X_train,y_train)
  y_pred=dt_1.predict(X_test)
  unp_acc=metrics.accuracy_score(y_test,y_pred)*100
  max_d=dt_1.tree_.max_depth
  sheet2.write(ct,3,max_d)
  acc_list=[]
  puma_2_df=df.copy()
  #puma_2_df=puma_2_df[(puma_2_df['race']==race)]#&(puma_2_df['PUMA']==puma)]
  #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
  feature_cols=list(df.columns)
  #print(feature_cols)
  feature_cols.remove('fnlwgt')
  #puma_1_df=prep_hist(puma_1_df)
  puma_2_df=prep_hist3(puma_2_df,feature_cols,'fnlwgt')
  feature_cols.remove('income_bi')
  #print(puma_1_df['COUNT'])
  puma_2_df=redact_counts_df(puma_2_df,6)
  puma_2_df=count_scale_df(puma_2_df)
  #feature_cols=['AGE','SEX','RACE','OWNERSHP']
  #print(puma_1_df.columns)
  X2=puma_2_df[feature_cols]
  y2=puma_2_df['income_bi']
  try:
    X2_train, X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3)
  except:
    print('blegh')
  acc_c=0
  i=0
  acc_list=[]
  while acc_c<unp_acc:
    i+=1
    #print(i)
    np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
    dt_1=dt_1.fit(X2_train,y2_train)
    y2_pred=dt_1.predict(X2_test)
    acc_list.append(metrics.accuracy_score(y2_test,y2_pred)*100)
    acc_c=acc_list[-1]
    if i>100:
      print('Too much')
      break
    #print(abs(unp_acc))
    #print(acc_c)
  #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
  sheet2.write(ct,4,2+acc_list.index(max(acc_list)))
  #R10
  puma_2_df=df.copy()
  #puma_2_df=puma_2_df[(puma_2_df['race']==race)]#&(puma_2_df['PUMA']==puma)]
  #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
  feature_cols=list(df.columns)
  #print(feature_cols)
  feature_cols.remove('fnlwgt')
  #puma_1_df=prep_hist(puma_1_df)
  puma_2_df=prep_hist3(puma_2_df,feature_cols,'fnlwgt')
  #print(puma_1_df['COUNT'])
  puma_2_df=redact_counts_df(puma_2_df,10)
  puma_2_df=count_scale_df(puma_2_df)
  feature_cols.remove('income_bi')
  #feature_cols=['AGE','SEX','RACE','OWNERSHP']
  #print(puma_1_df.columns)
  X2=puma_2_df[feature_cols]
  y2=puma_2_df['income_bi']
  try:
    X2_train, X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3)
  except:
    print('blegh')
  acc_c=0
  i=0
  acc_list=[]
  while acc_c<unp_acc:
    i+=1
    #print(i)
    np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
    dt_1=dt_1.fit(X2_train,y2_train)
    y2_pred=dt_1.predict(X2_test)
    acc_list.append(metrics.accuracy_score(y2_test,y2_pred)*100)
    acc_c=acc_list[-1]
    if i>100:
      print('Too much')
      break
    #print(abs(unp_acc))
    #print(acc_c)
  #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
  sheet2.write(ct,5,2+acc_list.index(max(acc_list)))
  #DP1
  puma_2_df=df.copy()
  #puma_2_df=puma_2_df[(puma_2_df['race']==race)]#&(puma_2_df['PUMA']==puma)]
  #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
  feature_cols=list(df.columns)
  #print(feature_cols)
  feature_cols.remove('fnlwgt')
  #puma_1_df=prep_hist(puma_1_df)
  puma_2_df=prep_hist3(puma_2_df,feature_cols,'fnlwgt')
  #print(puma_1_df['COUNT'])
  puma_2_df=dp_counts_df(puma_2_df,1)
  puma_2_df=count_scale_df(puma_2_df)
  feature_cols.remove('income_bi')
  #feature_cols=['AGE','SEX','RACE','OWNERSHP']
  #print(puma_1_df.columns)
  X2=puma_2_df[feature_cols]
  y2=puma_2_df['income_bi']
  try:
    X2_train, X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3)
  except:
    print('blegh')
  acc_c=0
  i=0
  acc_list=[]
  while acc_c<unp_acc:
    i+=1
    #print(i)
    np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
    dt_1=dt_1.fit(X2_train,y2_train)
    y2_pred=dt_1.predict(X2_test)
    acc_list.append(metrics.accuracy_score(y2_test,y2_pred)*100)
    acc_c=acc_list[-1]
    if i>100:
      print('Too much')
      break
    #print(abs(unp_acc))
    #print(acc_c)
  #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
  sheet2.write(ct,6,2+acc_list.index(max(acc_list)))
  #DP2
  puma_2_df=df.copy()
  #puma_2_df=puma_2_df[(puma_2_df['race']==race)]#&(puma_2_df['PUMA']==puma)]
  #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
  feature_cols=list(df.columns)
  #print(feature_cols)
  feature_cols.remove('fnlwgt')
  #puma_1_df=prep_hist(puma_1_df)
  puma_2_df=prep_hist3(puma_2_df,feature_cols,'fnlwgt')
  #print(puma_1_df['COUNT'])
  puma_2_df=dp_counts_df(puma_2_df,2)
  puma_2_df=count_scale_df(puma_2_df)
  feature_cols.remove('income_bi')
  #feature_cols=['AGE','SEX','RACE','OWNERSHP']
  #print(puma_1_df.columns)
  X2=puma_2_df[feature_cols]
  y2=puma_2_df['income_bi']
  try:
    X2_train, X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3)
  except:
    print('blegh')
  acc_c=0
  i=0
  acc_list=[]
  while acc_c<unp_acc:
    i+=1
    #print(i)
    np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
    dt_1=dt_1.fit(X2_train,y2_train)
    y2_pred=dt_1.predict(X2_test)
    acc_list.append(metrics.accuracy_score(y2_test,y2_pred)*100)
    acc_c=acc_list[-1]
    if i>100:
      print('Too much')
      break
    #print(abs(unp_acc))
    #print(acc_c)
  #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
  sheet2.write(ct,7,2+acc_list.index(max(acc_list)))
  #AM 6
  puma_2_df=df.copy()
  #puma_2_df=puma_2_df[(puma_2_df['race']==race)]#&(puma_2_df['PUMA']==puma)]
  #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
  feature_cols=list(df.columns)
  #print(feature_cols)
  feature_cols.remove('fnlwgt')
  #puma_1_df=prep_hist(puma_1_df)
  puma_2_df=prep_hist3(puma_2_df,feature_cols,'fnlwgt')
  #print(puma_1_df['COUNT'])
  puma_2_df=adjmerge_counts_df(puma_2_df,6)
  puma_2_df=count_scale_df(puma_2_df)
  feature_cols.remove('income_bi')
  #feature_cols=['AGE','SEX','RACE','OWNERSHP']
  #print(puma_1_df.columns)
  X2=puma_2_df[feature_cols]
  y2=puma_2_df['income_bi']
  try:
    X2_train, X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3)
  except:
    print('blegh')
  acc_c=0
  i=0
  acc_list=[]
  while acc_c<unp_acc:
    i+=1
    #print(i)
    np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
    dt_1=dt_1.fit(X2_train,y2_train)
    y2_pred=dt_1.predict(X2_test)
    acc_list.append(metrics.accuracy_score(y2_test,y2_pred)*100)
    acc_c=acc_list[-1]
    if i>100:
      print('Too much')
      break
    #print(abs(unp_acc))
    #print(acc_c)
  #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
  sheet2.write(ct,8,2+acc_list.index(max(acc_list)))
  #AM 10
  puma_2_df=df.copy()
  #puma_2_df=puma_2_df[(puma_2_df['race']==race)]#&(puma_2_df['PUMA']==puma)]
  #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
  feature_cols=list(df.columns)
  #print(feature_cols)
  feature_cols.remove('fnlwgt')
  #puma_1_df=prep_hist(puma_1_df)
  puma_2_df=prep_hist3(puma_2_df,feature_cols,'fnlwgt')
  #print(puma_1_df['COUNT'])
  puma_2_df=adjmerge_counts_df(puma_2_df,6)
  puma_2_df=count_scale_df(puma_2_df)
  feature_cols.remove('income_bi')
  #feature_cols=['AGE','SEX','RACE','OWNERSHP']
  #print(puma_1_df.columns)
  X2=puma_2_df[feature_cols]
  y2=puma_2_df['income_bi']
  try:
    X2_train, X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3)
  except:
    print('blegh')
  acc_c=0
  i=0
  acc_list=[]
  while acc_c<unp_acc:
    i+=1
    #print(i)
    np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
    dt_1=dt_1.fit(X2_train,y2_train)
    y2_pred=dt_1.predict(X2_test)
    acc_list.append(metrics.accuracy_score(y2_test,y2_pred)*100)
    acc_c=acc_list[-1]
    if i>100:
      print('Too much')
      break
    #print(abs(unp_acc))
    #print(acc_c)
  #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
  sheet2.write(ct,9,2+acc_list.index(max(acc_list)))
  #RM 6
  puma_2_df=df.copy()
  #puma_2_df=puma_2_df[(puma_2_df['race']==race)]#&(puma_2_df['PUMA']==puma)]
  #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
  feature_cols=list(df.columns)
  #print(feature_cols)
  feature_cols.remove('fnlwgt')
  #puma_1_df=prep_hist(puma_1_df)
  puma_2_df=prep_hist3(puma_2_df,feature_cols,'fnlwgt')
  #print(puma_1_df['COUNT'])
  puma_2_df=randmerge_counts_df(puma_2_df,6)
  puma_2_df=count_scale_df(puma_2_df)
  feature_cols.remove('income_bi')
  #feature_cols=['AGE','SEX','RACE','OWNERSHP']
  #print(puma_1_df.columns)
  X2=puma_2_df[feature_cols]
  y2=puma_2_df['income_bi']
  try:
    X2_train, X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3)
  except:
    print('blegh')
  acc_c=0
  i=0
  acc_list=[]
  while acc_c<unp_acc:
    i+=1
    #print(i)
    np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
    dt_1=dt_1.fit(X2_train,y2_train)
    y2_pred=dt_1.predict(X2_test)
    acc_list.append(metrics.accuracy_score(y2_test,y2_pred)*100)
    acc_c=acc_list[-1]
    if i>100:
      print('Too much')
      break
    #print(abs(unp_acc))
    #print(acc_c)
  #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
  sheet2.write(ct,10,2+acc_list.index(max(acc_list)))
  #RM 10
  puma_2_df=df.copy()
  #puma_2_df=puma_2_df[(puma_2_df['race']==race)]#&(puma_2_df['PUMA']==puma)]
  #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
  feature_cols=list(df.columns)
  #print(feature_cols)
  feature_cols.remove('fnlwgt')
  #puma_1_df=prep_hist(puma_1_df)
  puma_2_df=prep_hist3(puma_2_df,feature_cols,'fnlwgt')
  #print(puma_1_df['COUNT'])
  puma_2_df=randmerge_counts_df(puma_2_df,10)
  puma_2_df=count_scale_df(puma_2_df)
  feature_cols.remove('income_bi')
  #feature_cols=['AGE','SEX','RACE','OWNERSHP']
  #print(puma_1_df.columns)
  X2=puma_2_df[feature_cols]
  y2=puma_2_df['income_bi']
  try:
    X2_train, X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3)
  except:
    print('blegh')
  acc_c=0
  i=0
  acc_list=[]
  while acc_c<unp_acc:
    i+=1
    #print(i)
    np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
    dt_1=dt_1.fit(X2_train,y2_train)
    y2_pred=dt_1.predict(X2_test)
    acc_list.append(metrics.accuracy_score(y2_test,y2_pred)*100)
    acc_c=acc_list[-1]
    if i>100:
      print('Too much')
      break
    #print(abs(unp_acc))
    #print(acc_c)
  #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
  sheet2.write(ct,11,2+acc_list.index(max(acc_list)))
wbboh.save('RaceDTMatchUnpAdult.xls')
