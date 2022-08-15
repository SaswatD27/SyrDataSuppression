# -*- coding: utf-8 -*-
"""ACSAnalysis2.1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JmMiErJPuWX7AF40ahNYEu3hGaR_gMRm
"""

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
#import timeit
#import xlwt
#from xlwt import Workbook

"""##Analysis Tools"""

def zeroProb(p):
#                    this is the chance of "staying put at 0"
  if random.random()<(1.0 - p)/(1.0 + p):
    return 0
  else:
    return 1
    # Coin flip to determine if the result is negative or positive.
    # This only applies when we "leave 0."
def signProb():
  if random.random() < 0.5:
    return -1
  else:
    return 1

def twoSidedGeoDist(p):
#      (1) Did we "leave 0"? [Y=1|N=0]        (3) +/-
  return zeroProb(p) * np.random.geometric(1-p) * signProb()
def mech_disclaplace(eps):
  return twoSidedGeoDist(math.e**(-eps))
#--------------------

#----- Adding DP to Hist ----
def add_dp_hist(counts):
  dp_counts=[]
  for i in range(len(counts)):
    dp_counts.append(counts[i]+twoSidedGeoDist(math.e**(-eps)))
  return [abs(x) for x in dp_counts]
#----------------------------

#----- Multiple Runs of DP and extraction of worst runs ---

#----------------------------------------------------------

#----- Merge Columns ----------- (Probably useless, let's see)
def suppress_merge(df,colvec): #colvec: list of columns to merge
  data_suppressed=df.copy()
  data_suppressed=data_suppressed.drop(columns=colvec)
  #Then add a hybrid column with tuples
  #A column needs a name!
  colname=""
  for c in colvec:
    colname=colname+str(c)
  data_suppressed[str(colname)]=df[colvec].astype(str).agg('-'.join, axis=1)
  return data_suppressed
#-----------------------

#----- Binning ----------
def data_col_bin(df,col,wid):#col - index of the column, wid - desired width
  min_l=min(df[col])
  max_l=max(df[col])
  bins=[]
  for i in range(min_l//wid,max_l//wid):
    bin.append([i,i+wid])
  df_bin=df.copy()
  df_bin=df[col]
  return df_bin
#------------------------

#----- Merging Categories for a categorical attribute----

#--------------------------------------------------------

#----- Fairness Error ----
def fair_err(hist):
  return max(hist)-min(hist)
#-------------------------

#----- Randomly k anonymise by merging----
def merge_random(hist,col,k): #Merges a given bin/bucket with another bin
  hist1=list(hist)
  col2=0
  #print(hist)
  #print(hist[col]+hist[col2])
  while hist[col]+hist[col2]<k:
    col2=random.choice([*range(0,col),*range(col+1,len(hist))])
  hist1.append(hist[col]+hist[col2])
  del hist1[col]
  del hist1[col2]
  return hist1

def k_anonymise_rand(hist,k): #k anonymises a histogram with random bin merging
  hist=np.array(hist)
  #print(hist<k)
  while any(hist<k):
    i=np.where(hist<k)
    #print('i=',i[0][0])
    hist=np.array(merge_random(hist,i[0][0],k))
  return hist
#-------------------------------

#----- k anonymise and spread ---------
def spread_random(hist,cols,k): #Merges a given bin/bucket with another bin
  hist1=list(hist)
  col=cols[0]
  if col == 0:
    col2=1
  else:
    col2=0
  #print(hist)
  #print(hist[col]+hist[col2])
  while (hist[col]+hist[col2]<k) or (col2 in cols):
    col2=random.choice([*range(0,len(hist))])
    #print('lol')
  #print(col2)
  val=(hist[col]+hist[col2])/2
  hist1[col2]=val
  hist1[col]=val
  return hist1

def spread_k_anonymise(hist,k): #k anonymises a histogram with random bin merging
  hist=np.array(hist)
  #print(hist<k)
  while any(hist<k):
    #print('lel')
    i=np.where(hist<k)
    #print(i)
    #print('i=',i[0][0])
    hist=np.array(spread_random(hist,i[0],k))
  return hist.astype(int)
#--------------------------------------

#----- Suppress by Omission ----
def supp_omit(hist, thres):
  hist1=list(hist)
  for i in range(len(hist1)):
    if hist1[i]<thres:
      hist1[i]=0
  return hist1
#-------------------------------

#----- k anon adj merge ------#incomplete
def spread_bins(hist,col1,col2):
  hist1=list(hist)
  hist1[col1]=(hist[col1]+hist[col2])/2
  hist1[col2]=float(hist1[col1])
  return hist1

def spread_adjacent(hist,col,k): #Merges a given bin/bucket with adjacent bins
  hist1=list(hist)
  if col==len(hist)-1:
    hist1=list(spread_bins(hist1,col,col-1))
  elif col==0:
    hist1=list(spread_bins(hist1,col,col+1))
  else:
    if hist1[col-1]>hist1[col+1]:
      hist1=list(spread_bins(hist1,col,col-1))
    else:
      hist1=list(spread_bins(hist1,col,col+1))
  return hist1

def adj_k_anonymise(hist,k):
  hist=np.array(hist)
  l=0
  while any(hist<k):
    l+=1
    print(l)
    i=np.where(hist<k)
    print(i)
    for j in i[0]:
      spread_adjacent(hist,j,k)
  return hist.astype(int)
#-----------------------------

#----- Times DP is beaten ----
#Perhaps write it in the notebook itself
#-----------------------------

"""Correlation. k as percentiles."""

def five_num_summ(hist):
  list=[]
  hist1=np.array(hist)
  list.append(min(hist1))
  list.append(np.percentile(hist,25))
  list.append(np.percentile(hist,50))
  list.append(np.percentile(hist,75))
  list.append(max(hist))
  return list

def rand_merge_errs(counts_1,k):
  errs_randomk_95_counts_1=[]
  fair_errs_randomk_95_counts_1=[]
  for i in range(1000):
    k_counts_1=syr.spread_k_anonymise(counts_1,k)
    #print(k_counts_1)
    errs_randomk_95_counts_1.append(sum(abs(np.array(k_counts_1)-counts_1)))
    fair_errs_randomk_95_counts_1.append(max(abs(np.array(k_counts_1)-counts_1))-min(abs(np.array(k_counts_1)-counts_1)))
  return errs_randomk_95_counts_1, fair_errs_randomk_95_counts_1

def dp_errs(counts_1,eps):
  errs_dp_counts_1=[]
  fair_errs_dp_counts_1=[]
  for i in range(1000):
    dp_counts_1=syr.add_dp_hist(counts_1,eps)
    #print(k_counts_1)
    errs_dp_counts_1.append(sum(abs(np.array(dp_counts_1)-counts_1)))
    fair_errs_dp_counts_1.append(max(abs(np.array(dp_counts_1)-counts_1))-min(abs(np.array(dp_counts_1)-counts_1)))
  return errs_dp_counts_1, fair_errs_dp_counts_1

"""# Import Dataset

Packages for rep on a map. Join with another table for counties. Maybe try for different PUMAs
"""

#import dataset - tx
df=pd.read_csv("tx_acs_2019.csv")

"""#Marginal Query 1"""

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
    for i in range(row['COUNT']):
      hist_r=hist_r.append(row)
  return hist_r
#
def redact_counts_df(hist,k): #Outputs redact-suppressed histogram
  hist1=hist.copy()
  counts_1=list(hist1['COUNT'])
  #print("counts:",counts_1)
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


"""# Query Function"""

def marg_query(hist, fnames): #count query; pass hist_r into this
  return hist.groupby(fnames).count()['COUNT'].values.tolist()

def marg_query_2(hist,fnames):
  list0=list(hist[fnames[0]].unique())
  list1=list(hist[fnames[1]].unique())
  counts=[]
  for i in list0:
    for j in list1:
      counts.append(sum(hist[(hist[fnames[0]]==i) & (hist[fnames[1]]==j)]['COUNT']))
  return counts

"""# DP"""
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

def am_errs2(hist,fnames,eps): #return l1errs, fairerrs 
  #hist=prep_hist(df)
  #hist_r=count_scale_df(hist)
  l1errs=0
  fairerrs=0
  #for iter in range(100):
  #print(iter)
  hist_dp=adjmerge_counts_df(hist,eps)
  #hist_dp_r=count_scale_df(hist_dp)
  dp_resp_vec=np.array(marg_query_2(hist_dp,fnames))
  dp_raw_vec=np.array(marg_query_2(hist,fnames))
  l1errs=(sum(abs(dp_resp_vec-dp_raw_vec)))
  fairerrs=(max(abs(dp_resp_vec-dp_raw_vec))-min(abs(dp_resp_vec-dp_raw_vec)))
  return l1errs, fairerrs

def red_errs2(hist,fnames,eps): #return l1errs, fairerrs 
  #hist=prep_hist(df)
  #hist_r=count_scale_df(hist)
  l1errs=0
  fairerrs=0
  #for iter in range(100):
  #print(iter)
  hist_dp=redact_counts_df(hist,eps)
  #hist_dp_r=count_scale_df(hist_dp)
  dp_resp_vec=np.array(marg_query_2(hist_dp,fnames))
  dp_raw_vec=np.array(marg_query_2(hist,fnames))
  l1errs=(sum(abs(dp_resp_vec-dp_raw_vec)))
  fairerrs=(max(abs(dp_resp_vec-dp_raw_vec))-min(abs(dp_resp_vec-dp_raw_vec)))
  return l1errs, fairerrs

def dp_errs(hist,fnames,eps): #return l1errs, fairerrs 
  #hist=prep_hist(df)
  hist_r=count_scale_df(hist)
  l1errs=[]
  fairerrs=[]
  for iter in range(1000):
    print(iter)
    hist_dp=dp_counts_df(hist,eps)
    hist_dp_r=count_scale_df(hist_dp)
    dp_resp_vec=np.array(marg_query(hist_dp_r,fnames))
    dp_raw_vec=np.array(marg_query(hist_r,fnames))
    l1errs.append(sum(abs(dp_resp_vec-dp_raw_vec)))
    fairerrs.append(max(abs(dp_resp_vec-dp_raw_vec))-min(abs(dp_resp_vec-dp_raw_vec)))
  return l1errs, fairerrs

def dp_errs2(hist,fnames,eps): #return l1errs, fairerrs 
  #hist=prep_hist(df)
  #hist_r=count_scale_df(hist)
  l1errs=[]
  fairerrs=[]
  for iter in range(1000):
    print(iter)
    hist_dp=dp_counts_df(hist,eps)
    #hist_dp_r=count_scale_df(hist_dp)
    dp_resp_vec=np.array(marg_query_2(hist_dp,fnames))
    dp_raw_vec=np.array(marg_query_2(hist,fnames))
    l1errs.append(sum(abs(dp_resp_vec-dp_raw_vec)))
    fairerrs.append(max(abs(dp_resp_vec-dp_raw_vec))-min(abs(dp_resp_vec-dp_raw_vec)))
  return l1errs, fairerrs
'''
#for eps in [1,2,3,5]:
hist1=prep_hist(df)
fnameses=[['AGE','RACE'],['RACE','SEX'],['RACE','OWNERSHP'],['SEX','OWNERSHP']]
#fnames=['AGE','OWNERSHP']
for fnames in fnameses:
  l1errs1,fairerrs1=dp_errs2(hist1,fnames,1)
  l1errs2,fairerrs2=dp_errs2(hist1,fnames,2)
  l1errs3,fairerrs3=dp_errs2(hist1,fnames,3)
  l1errs5,fairerrs5=dp_errs2(hist1,fnames,5)
  sns.boxplot(data=pd.DataFrame({'1':l1errs1, '2':l1errs2, '3':l1errs3, '5':l1errs5})).set(title='DP L1 '+str(fnames))
  plt.savefig('DP_L1_'+str(fnames)+'.pdf',format='pdf',dpi=300)
  sns.boxplot(data=pd.DataFrame({'1':fairerrs1, '2':fairerrs2, '3':fairerrs3, '5':fairerrs5})).set(title='DP Fair '+str(fnames))
  plt.clf()
  plt.savefig('DP_Fair_'+str(fnames)+'.pdf',format='pdf',dpi=300)
  plt.clf()
'''
def randmerge_errs(hist,fnames,k): #return l1errs, fairerrs 
  #hist=prep_hist(df)
  hist_r=count_scale_df(hist)
  l1errs=[]
  fairerrs=[]
  for iter in range(1000):
    hist_dp=randmerge_counts_df(hist,k)
    hist_dp_r=count_scale_df(hist_dp)
    dp_resp_vec=np.array(marg_query(hist_dp_r,fnames))
    dp_raw_vec=np.array(marg_query(hist_r,fnames))
    l1errs.append(sum(abs(dp_resp_vec-dp_raw_vec)))
    fairerrs.append(max(abs(dp_resp_vec-dp_raw_vec))-min(abs(dp_resp_vec-dp_raw_vec)))
  return l1errs, fairerrs

def randmerge_errs2(hist,fnames,k): #return l1errs, fairerrs 
  #hist=prep_hist(df)
  #hist_r=count_scale_df(hist)
  l1errs=[]
  fairerrs=[]
  for iter in range(1000):
    #print(iter)
    hist_dp=randmerge_counts_df(hist,k)
    #hist_dp_r=count_scale_df(hist_dp)
    dp_resp_vec=np.array(marg_query_2(hist_dp,fnames))
    dp_raw_vec=np.array(marg_query_2(hist,fnames))
    l1errs.append(sum(abs(dp_resp_vec-dp_raw_vec)))
    fairerrs.append(max(abs(dp_resp_vec-dp_raw_vec))-min(abs(dp_resp_vec-dp_raw_vec)))
  return l1errs, fairerrs
'''
fnameses=[['AGE','RACE'],['RACE','SEX'],['RACE','OWNERSHP'],['SEX','OWNERSHP']]
#fnames=['AGE','OWNERSHP']
for fnames in fnameses:
#fnames=['AGE','OWNERSHP']
  l1errsx2,fairerrsx2=randmerge_errs2(hist1,fnames,2)
  l1errsx3,fairerrsx3=randmerge_errs2(hist1,fnames,3)
  l1errsx6,fairerrsx6=randmerge_errs2(hist1,fnames,6)
  sns.boxplot(data=pd.DataFrame({'2':l1errsx2, '3':l1errsx3, '6':l1errsx6})).set(title='Rand Merge L1 '+str(fnames))
  plt.savefig('RM_L1_'+str(fnames)+'.pdf',format='pdf',dpi=300)
  plt.clf()
  sns.boxplot(data=pd.DataFrame({'2':fairerrsx2, '3':fairerrsx3, '6':fairerrsx6})).set(title='Rand Merge Fair '+str(fnames))
  plt.savefig('RM_Fair_'+str(fnames)+'.pdf',format='pdf',dpi=300)
  plt.clf()
'''
"""#Per PUMA"""

def perdbanalysis():
  df_names=["outlier_acs_2019.csv","tx_acs_2019.csv","ma_acs_2019.csv"]
  for df_name in df_names:
    df=pd.read_csv(df_name)
    #pumas=list(df['PUMA'].unique())
    #for puma in pumas:
      #for eps in [1,2,3,5]:
    hist1=prep_hist(df)
    fnameses=[['AGE','OWNERSHP'],['AGE','SEX'],['AGE','RACE'],['RACE','SEX'],['RACE','OWNERSHP'],['SEX','OWNERSHP']]
      #fnames=['AGE','OWNERSHP']
    for fnames in fnameses:
      #redact
      l1errs6, fairerrs6=red_errs2(hist1,fnames,6)
      l1errs10, fairerrs10=red_errs2(hist1,fnames,10)
      print('\nRedact\nl1 errors for',str(fnames),' w/ k=6:',l1errs6,'w/ k=10: ',l1errs10,'\nFairness errors for ',str(fnames),' w/ k=6: ',fairerrs6,' w/ k=10: ',fairerrs10)
      #l1errs1,fairerrs1=dp_errs2(hist1,fnames,1)
      #l1errs2,fairerrs2=dp_errs2(hist1,fnames,2)
      #l1errs3,fairerrs3=dp_errs2(hist1,fnames,3)
      #l1errs5,fairerrs5=dp_errs2(hist1,fnames,5)
      #sns.boxplot(data=pd.DataFrame({'1':l1errs1, '2':l1errs2, '3':l1errs3, '5':l1errs5})).set(title='DP L1 '+str(fnames))
      #plt.savefig(str(df_name)+'1 DP_L1_'+str(fnames)+'.pdf',format='pdf',dpi=300)
      #plt.clf()
      #sns.boxplot(data=pd.DataFrame({'1':fairerrs1, '2':fairerrs2, '3':fairerrs3, '5':fairerrs5})).set(title='DP Fair '+str(fnames))
      #plt.savefig(str(df_name)+'1 DP_Fair_'+str(fnames)+'.pdf',format='pdf',dpi=300)
      #plt.clf()
      #fnameses=[['AGE','RACE'],['RACE','SEX'],['RACE','OWNERSHP'],['SEX','OWNERSHP']]
      #fnames=['AGE','OWNERSHP']
    for fnames in fnameses:
      l1errsx6, fairerrsx6=am_errs2(hist1,fnames,6)
      l1errsx10, fairerrsx10=am_errs2(hist1,fnames,10)
      print('\nAM\nl1 errors for',str(fnames),' w/ k=6:',l1errsx6,'w/ k=10: ',l1errsx10,'\n Fairness errors for ',str(fnames),' w/ k=6: ',fairerrsx6,' \w k=10: ',fairerrsx10)
      #l1errs1,fairerrs1=dp_errs2(hist1,fnames,1)
      #fnames=['AGE','OWNERSHP']
      #l1errsx2,fairerrsx2=randmerge_errs2(hist1,fnames,2)
      #l1errsx3,fairerrsx3=randmerge_errs2(hist1,fnames,3)
      #l1errsx6,fairerrsx6=randmerge_errs2(hist1,fnames,6)
      #sns.boxplot(data=pd.DataFrame({'2':l1errsx2, '3':l1errsx3, '6':l1errsx6})).set(title='Rand Merge L1 '+str(fnames))
      #plt.savefig(str(df_name)+'1 RM_L1_'+str(fnames)+'.pdf',format='pdf',dpi=300)
      #plt.clf()
      #sns.boxplot(data=pd.DataFrame({'2':fairerrsx2, '3':fairerrsx3, '6':fairerrsx6})).set(title='Rand Merge Fair '+str(fnames))
      #plt.savefig(str(df_name)+'1 RM_Fair_'+str(fnames)+'.pdf',format='pdf',dpi=300)
      #plt.clf()

def perpumaanalysis():
  df_names=["outlier_acs_2019.csv","tx_acs_2019.csv","ma_acs_2019.csv"]
  for df_name in df_names:
    df=pd.read_csv(df_name)
    pumas=list(df['PUMA'].unique())
    l1errs1tot=np.zeros(1000)
    l1errs2tot=np.zeros(1000)
    l1errs3tot=np.zeros(1000)
    l1errs5tot=np.zeros(1000)
    fairerrs1tot=np.zeros(1000)
    fairerrs2tot=np.zeros(1000)
    fairerrs3tot=np.zeros(1000)
    fairerrs5tot=np.zeros(1000)
    ###
    l1errsx2tot=np.zeros(1000)
    l1errsx3tot=np.zeros(1000)
    l1errsx6tot=np.zeros(1000)
    fairerrsx2tot=np.zeros(1000)
    fairerrsx3tot=np.zeros(1000)
    fairerrsx6tot=np.zeros(1000)
    for puma in pumas:
    #for eps in [1,2,3,5]:
      hist1=prep_hist(df)
      fnameses=[['AGE','OWNERSHP'],['AGE','SEX'],['AGE','RACE'],['RACE','SEX'],['RACE','OWNERSHP'],['SEX','OWNERSHP']]
      #fnames=['AGE','OWNERSHP']
      for fnames in fnameses:
        l1errs1,fairerrs1=dp_errs2(hist1,fnames,1)
        l1errs2,fairerrs2=dp_errs2(hist1,fnames,2)
        l1errs3,fairerrs3=dp_errs2(hist1,fnames,3)
        l1errs5,fairerrs5=dp_errs2(hist1,fnames,5)
        l1errs1tot=l1errs1tot+np.array(l1errs1)
        l1errs2tot=l1errs2tot+np.array(l1errs2)
        l1errs3tot=l1errs3tot+np.array(l1errs3)
        l1errs5tot=l1errs5tot+np.array(l1errs5)
        fairerrs1tot=fairerrs1tot+np.array(fairerrs1)
        fairerrs2tot=fairerrs2tot+np.array(fairerrs2)
        fairerrs3tot=fairerrs3tot+np.array(fairerrs3)
        fairerrs5tot=fairerrs5tot+np.array(fairerrs5)
        #sns.boxplot(data=pd.DataFrame({'1':l1errs1, '2':l1errs2, '3':l1errs3, '5':l1errs5})).set(title=str(puma)+' DP L1 '+str(fnames))
        #plt.savefig('DP_L1_'+str(fnames)+'.pdf',format='pdf',dpi=300)
        #sns.boxplot(data=pd.DataFrame({'1':fairerrs1, '2':fairerrs2, '3':fairerrs3, '5':fairerrs5})).set(title=str(puma)+' DP Fair '+str(fnames))
        #plt.savefig('DP_Fair_'+str(fnames)+'.pdf',format='pdf',dpi=300)
        sns.boxplot(data=pd.DataFrame({'1':l1errs1tot/6, '2':l1errs2tot/6, '3':l1errs3/6, '5':l1errs5/6})).set(title=str(puma)+' DP L1')
        plt.savefig('DP_L1_'+str(puma)+'.pdf',format='pdf',dpi=300)
        plt.clf()
        sns.boxplot(data=pd.DataFrame({'1':fairerrs1tot/6, '2':fairerrs2tot/6, '3':fairerrs3tot/6, '5':fairerrs5tot/6})).set(title=str(puma)+' DP Fair')  
        plt.savefig('DP_Fair_'+str(puma)+'.pdf',format='pdf',dpi=300)
        plt.clf()
	#fnameses=[['AGE','RACE'],['RACE','SEX'],['RACE','OWNERSHP'],['SEX','OWNERSHP']]
        #fnames=['AGE','OWNERSHP']
      for fnames in fnameses:
      #fnames=['AGE','OWNERSHP']
        l1errsx2,fairerrsx2=randmerge_errs2(hist1,fnames,2)
        l1errsx3,fairerrsx3=randmerge_errs2(hist1,fnames,3)
        l1errsx6,fairerrsx6=randmerge_errs2(hist1,fnames,6)
        l1errsx2tot=l1errsx2tot+np.array(l1errsx2)
        l1errsx3tot=l1errsx3tot+np.array(l1errsx3)
        l1errsx6tot=l1errsx6tot+np.array(l1errsx6)
        fairerrsx2tot=fairerrsx2tot+np.array(fairerrsx2)
        fairerrsx3tot=fairerrsx3tot+np.array(fairerrsx3)
        fairerrsx6tot=fairerrsx6tot+np.array(fairerrsx6)
        #sns.boxplot(data=pd.DataFrame({'2':l1errsx2, '3':l1errsx3, '6':l1errsx6})).set(title=str(puma)+' Rand Merge L1 '+str(fnames))
        #plt.savefig('RM_L1_'+str(fnames)+'.pdf',format='pdf',dpi=300)
	#plt.clf()
        #sns.boxplot(data=pd.DataFrame({'2':fairerrsx2, '3':fairerrsx3, '6':fairerrsx6})).set(title=str(puma)+' Rand Merge Fair '+str(fnames))
        #plt.savefig('RM_Fair_'+str(fnames)+'.pdf',format='pdf',dpi=300)
	#plt.clf()
        sns.boxplot(data=pd.DataFrame({'2':l1errsx2tot/6, '3':l1errsx3tot/6, '6':l1errsx6tot/6})).set(title=str(puma)+' Rand Merge L1')
        plt.savefig('RM_L1_'+str(puma)+'.pdf',format='pdf',dpi=300)
        plt.clf()
        sns.boxplot(data=pd.DataFrame({'2':fairerrsx2tot/6, '3':fairerrsx3tot/6, '6':fairerrsx6tot/6})).set(title=str(puma)+' Rand Merge Fair')
        plt.savefig('RM_Fair_'+str(puma)+'.pdf',format='pdf',dpi=300)
        plt.clf()
perdbanalysis()
#perpumaanalysis()
