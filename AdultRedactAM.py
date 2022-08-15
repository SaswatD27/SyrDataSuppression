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
#create function to replicate rows based on GQ value
def count_scale_df(hist):
  hist_r=hist.copy()
  for index, row in hist.iterrows():
    #print(row['COUNT'])
    for i in range(int(row['COUNT'])):
      hist_r=hist_r.append(row)
  return hist_r
#
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

def marg_query_2(hist,fnames):
  list0=list(hist[fnames[0]].unique())
  list1=list(hist[fnames[1]].unique())
  counts=[]
  for i in list0:
    for j in list1:
      counts.append(sum(hist[(hist[fnames[0]]==i) & (hist[fnames[1]]==j)]['COUNT']))
  return counts

def adjmerge_errs2(hist,fnames,k): #return l1errs, fairerrs 
  #hist=prep_hist(df)
  #hist_r=count_scale_df(hist)
  l1errs=[]
  fairerrs=[]
  #for iter in range(1000):
  #print(iter)
  hist_dp=adjmerge_counts_df(hist,k)
  #hist_dp_r=count_scale_df(hist_dp)
  dp_resp_vec=np.array(marg_query_2(hist_dp,fnames))
  dp_raw_vec=np.array(marg_query_2(hist,fnames))
  l1errs=(sum(abs(dp_resp_vec-dp_raw_vec)))
  fairerrs=(max(abs(dp_resp_vec-dp_raw_vec))-min(abs(dp_resp_vec-dp_raw_vec)))
  return l1errs, fairerrs

def redact_errs2(hist,fnames,k): #return l1errs, fairerrs 
  #hist=prep_hist(df)
  #hist_r=count_scale_df(hist)
  l1errs=[]
  fairerrs=[]
  #for iter in range(1000):
  #print(iter)
  hist_dp=redact_counts_df(hist,k)
  #hist_dp_r=count_scale_df(hist_dp)
  dp_resp_vec=np.array(marg_query_2(hist_dp,fnames))
  dp_raw_vec=np.array(marg_query_2(hist,fnames))
  l1errs=(sum(abs(dp_resp_vec-dp_raw_vec)))
  fairerrs=(max(abs(dp_resp_vec-dp_raw_vec))-min(abs(dp_resp_vec-dp_raw_vec)))
  return l1errs, fairerrs
df=pd.read_csv('adult.csv')
#pumas=list(df['PUMA'].unique())
fnames=list(df.columns)
fnames.remove('fnlwgt')
hist1=prep_hist3(df,fnames,'fnlwgt')
#for puma in pumas:  
  #puma=pumas[1]
l1errsam2tot=0
l1errsam3tot=0
l1errsam6tot=0
fairerrsam2tot=0
fairerrsam3tot=0
fairerrsam6tot=0
fnameses=[['race','gender'],['race','age'],['race','education'],['gender','age'],['gender','education'],['age','education']]
for fnames in fnameses:
  #fnames=['AGE','OWNERSHP']
  l1errsam2,fairerrsam2=adjmerge_errs2(hist1,fnames,2)
  l1errsam3,fairerrsam3=adjmerge_errs2(hist1,fnames,3)
  l1errsam6,fairerrsam6=adjmerge_errs2(hist1,fnames,6)
  l1errsam2tot+=l1errsam2
  l1errsam3tot+=l1errsam3
  l1errsam6tot+=l1errsam6
  fairerrsam2tot+=fairerrsam2
  fairerrsam3tot+=fairerrsam3
  fairerrsam6tot+=fairerrsam6
  #print('PUMA - ',str(puma))
  print('\nAdj Merge L1 Err for ',str(fnames),' for k = 2 is ', l1errsam2tot/6)
  print('\nAdj Merge L1 Err for ',str(fnames),' for k = 3 is ', l1errsam3tot/6)
  print('\nAdj Merge L1 Err for ',str(fnames),' for k = 6 is ', l1errsam6tot/6)
  print('\nAdj Merge Fair Err for ',str(fnames),' for k = 2 is ', fairerrsam2tot/6)
  print('\nAdj Merge Fair Err for ',str(fnames),' for k = 3 is ', fairerrsam3tot/6)
  print('\nAdj Merge Fair Err for ',str(fnames),' for k = 6 is ', fairerrsam6tot/6)

l1errsam2tot=0
l1errsam3tot=0
l1errsam6tot=0
fairerrsam2tot=0
fairerrsam3tot=0
fairerrsam6tot=0
for fnames in fnameses:
  #fnames=['AGE','OWNERSHP']
  l1errsam2,fairerrsam2=redact_errs2(hist1,fnames,2)
  l1errsam3,fairerrsam3=redact_errs2(hist1,fnames,3)
  l1errsam6,fairerrsam6=redact_errs2(hist1,fnames,6)
  l1errsam2tot+=l1errsam2
  l1errsam3tot+=l1errsam3
  l1errsam6tot+=l1errsam6
  fairerrsam2tot+=fairerrsam2
  fairerrsam3tot+=fairerrsam3
  fairerrsam6tot+=fairerrsam6
  #print('PUMA - ',str(puma))
  print('\nRedact L1 Err for ',str(fnames),' for k = 2 is ', l1errsam2tot/6)
  print('\nRedact L1 Err for ',str(fnames),' for k = 3 is ', l1errsam3tot/6)
  print('\nRedact L1 Err for ',str(fnames),' for k = 6 is ', l1errsam6tot/6)
  print('\nRedact Fair Err for ',str(fnames),' for k = 2 is ', fairerrsam2tot/6)
  print('\nRedact Fair Err for ',str(fnames),' for k = 3 is ', fairerrsam3tot/6)
  print('\nRedact Fair Err for ',str(fnames),' for k = 6 is ', fairerrsam6tot/6)
