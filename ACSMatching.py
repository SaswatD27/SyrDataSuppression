#import libraries
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
#SOLID! Compares optimal heights for each race over different methods
df1=pd.read_csv("tx_acs_2019.csv")
pumas=list(df1['PUMA'].unique())
k=6
ct=0
sheet2=wbboh.add_sheet('TX')
#sheet3=wbb.add_sheet('TX Feat Imp - Redact 10')
sheet2.write(0, 0, 'PUMAs')
sheet2.write(0, 1, 'DT Race')
sheet2.write(0, 2, 'PUMA')
sheet2.write(0, 3, 'Unperturbed Height')
sheet2.write(0, 4, 'Randmerge 6')
sheet2.write(0, 5, 'Randmerge 10')
#sheet2.write(0, 6, 'RM 6')
#sheet2.write(0, 7, 'RM 10')
#pumas=[pumas[1]]
for puma in pumas:
  for race in range(1,8):
    ct+=1
    sheet2.write(ct,2,puma)
    print('\nRace - ',race)
    sheet2.write(ct,0,'TX')
    sheet2.write(ct,1,race)
    #print('\nRACE = ',race)
    puma_1_df=df1.copy()
    puma_1_df=puma_1_df[(puma_1_df['RACE']==race)&(puma_1_df['PUMA']==puma)]
    #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
    feature_cols=list(puma_1_df.columns)
    #print(feature_cols)
    feature_cols.remove('INCTOT')
    feature_cols.remove('PUMA')
    feature_cols.remove('INDNAICS')
    feature_cols.remove('Unnamed: 0')
    #puma_1_df=prep_hist(puma_1_df)
    puma_1_df=prep_hist2(puma_1_df,feature_cols)
    #print(puma_1_df['COUNT'])
    #puma_1_df=redact_counts_df(puma_1_df,k)
    puma_1_df=count_scale_df(puma_1_df)
    feature_cols.remove('INCTOT_DECILE')
    #feature_cols=['AGE','SEX','RACE','OWNERSHP']
    #print(puma_1_df.columns)
    X=puma_1_df[feature_cols]
    y=puma_1_df['INCTOT_DECILE']
    #print('\n',feature_cols)
    np.random.seed(0)
    try:
      X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    except:
    #  sheet2.write(ct,2,'Beep')
      #print('Not enough individuals')
      continue
    #print(y_test)
    #np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0)#, max_depth=2)
    dt_1=dt_1.fit(X_train,y_train)
    y_pred=dt_1.predict(X_test)
    unp_acc=metrics.accuracy_score(y_test,y_pred)*100
    max_d=dt_1.tree_.max_depth
    sheet2.write(ct,3,max_d)
    #acc_list=[]
    ht=0
    for _ in range(100):  
      puma_2_df=pd.read_csv("tx_acs_2019.csv")
      puma_2_df=puma_2_df[(puma_2_df['RACE']==race)&(puma_2_df['PUMA']==puma)]
      #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
      feature_cols=list(puma_2_df.columns)
      #print(feature_cols)
      feature_cols.remove('INCTOT')
      feature_cols.remove('PUMA')
      feature_cols.remove('INDNAICS')
      feature_cols.remove('Unnamed: 0')
      #puma_1_df=prep_hist(puma_1_df)
      puma_2_df=prep_hist2(puma_2_df,feature_cols)
      #print(puma_1_df['COUNT'])
      if len(puma_2_df)<2:
    	  continue
      puma_2_df=randmerge_counts_df(puma_2_df,6)
      puma_2_df=count_scale_df(puma_2_df)
      feature_cols.remove('INCTOT_DECILE')
      #feature_cols=['AGE','SEX','RACE','OWNERSHP']
      #print(puma_1_df.columns)
      X2=puma_2_df[feature_cols]
      y2=puma_2_df['INCTOT_DECILE']
      try:
        X2_train, X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3)
      except:
        continue
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
      ht+=2+acc_list.index(max(acc_list))
    ht/=100
    sheet2.write(ct,4,ht)
    #R10
    ht10=0
    for _ in range(100):  
      puma_2_df=pd.read_csv("tx_acs_2019.csv")
      puma_2_df=puma_2_df[(puma_2_df['RACE']==race)&(puma_2_df['PUMA']==puma)]
      #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
      feature_cols=list(puma_2_df.columns)
      #print(feature_cols)
      feature_cols.remove('INCTOT')
      feature_cols.remove('PUMA')
      feature_cols.remove('INDNAICS')
      feature_cols.remove('Unnamed: 0')
      #puma_1_df=prep_hist(puma_1_df)
      puma_2_df=prep_hist2(puma_2_df,feature_cols)
      #print(puma_1_df['COUNT'])
      if len(puma_2_df)<2:
    	  continue
      puma_2_df=randmerge_counts_df(puma_2_df,10)
      puma_2_df=count_scale_df(puma_2_df)
      feature_cols.remove('INCTOT_DECILE')
      #feature_cols=['AGE','SEX','RACE','OWNERSHP']
      #print(puma_1_df.columns)
      X2=puma_2_df[feature_cols]
      y2=puma_2_df['INCTOT_DECILE']
      try:
        X2_train, X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3)
      except:
        continue
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
      ht10+=2+acc_list.index(max(acc_list))
    ht10/=100
    sheet2.write(ct,5,ht10)
wbboh.save('ACSMatchingRM.xls')
