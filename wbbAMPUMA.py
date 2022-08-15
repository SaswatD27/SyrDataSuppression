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
import xlwt
from xlwt import Workbook
pumaind=0
"""##Analysis Tools"""
wbb=Workbook()
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
  for i in range(100):
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

def adjmerge_counts_df(hist,k): #Outputs redact-suppressed histogram
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

def dp_errs(hist,fnames,eps): #return l1errs, fairerrs 
  #hist=prep_hist(df)
  hist_r=count_scale_df(hist)
  l1errs=[]
  fairerrs=[]
  for iter in range(100):
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
  for iter in range(100):
    print(iter)
    hist_dp=dp_counts_df(hist,eps)
    #hist_dp_r=count_scale_df(hist_dp)
    dp_resp_vec=np.array(marg_query_2(hist_dp,fnames))
    dp_raw_vec=np.array(marg_query_2(hist,fnames))
    l1errs.append(sum(abs(dp_resp_vec-dp_raw_vec)))
    fairerrs.append(max(abs(dp_resp_vec-dp_raw_vec))-min(abs(dp_resp_vec-dp_raw_vec)))
  return l1errs, fairerrs

def red_errs2(hist,fnames,eps): 
  #hist=prep_hist(df)
  #hist_r=count_scale_df(hist)
  hist_dp=redact_counts_df(hist,eps)
  l1errs=[]
  fairerrs=[]
  dp_resp_vec=np.array(marg_query_2(hist_dp,fnames))
  dp_raw_vec=np.array(marg_query_2(hist,fnames))
  l1errs.append=(sum(abs(dp_resp_vec-dp_raw_vec)))
  fairerrs.append=(max(abs(dp_resp_vec-dp_raw_vec))-min(abs(dp_resp_vec-dp_raw_vec)))
  #for iter in range(100):
  #  print(iter)
  #  hist_dp=redact_counts_df(hist,eps)
  #  #hist_dp_r=count_scale_df(hist_dp)
  #  dp_resp_vec=np.array(marg_query_2(hist_dp,fnames))
  #  dp_raw_vec=np.array(marg_query_2(hist,fnames))
  #  l1errs.append(sum(abs(dp_resp_vec-dp_raw_vec)))
  #  fairerrs.append(max(abs(dp_resp_vec-dp_raw_vec))-min(abs(dp_resp_vec-dp_raw_vec)))
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
  for iter in range(100):
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
  for iter in range(100):
    #print(iter)
    hist_dp=randmerge_counts_df(hist,k)
    #hist_dp_r=count_scale_df(hist_dp)
    dp_resp_vec=np.array(marg_query_2(hist_dp,fnames))
    dp_raw_vec=np.array(marg_query_2(hist,fnames))
    l1errs.append(sum(abs(dp_resp_vec-dp_raw_vec)))
    fairerrs.append(max(abs(dp_resp_vec-dp_raw_vec))-min(abs(dp_resp_vec-dp_raw_vec)))
  return l1errs, fairerrs

df1=pd.read_csv("tx_acs_2019.csv")
pumas=list(df1['PUMA'].unique())
ct=0
k=6
sheet2=wbb.add_sheet('TX - AM 6')
sheet3=wbb.add_sheet('TX Feat Imp - AM 6')
sheet2.write(0, 0, 'PUMAs')
sheet2.write(0, 1, 'Race')
sheet2.write(0, 2, 'Accuracy')
sheet2.write(0, 3, 'Height')
sheet2.write(0, 4, 'Opt Ht')
sheet2.write(0, 5, 'Opt Acc')
sheet2.write(0, 6, 'Max Ht 10')
sheet2.write(0, 7, 'Max Ht 20')
sheet2.write(0, 8, 'Pop %')
sheet3.write(0,0,'PUMAs')
sheet3.write(0,1,'Race')
sheet3.write(0,2,'GQ')
sheet3.write(0,3,'OWNERSHP')
sheet3.write(0,4,'FAMSIZE')
sheet3.write(0,5,'NCHILD')
sheet3.write(0,6,'AGE')
sheet3.write(0,7,'SEX')
sheet3.write(0,8,'MARST')
sheet3.write(0,9,'RACE')
sheet3.write(0,10,'HISPAN')
sheet3.write(0,11,'EDUC')
sheet3.write(0,12,'SECTOR')
sheet3.write(0,13,'POVERTY')
sheet3.write(0,14,'DENSITY')
sheet3.write(0,15,'VETDISAB')
sheet3.write(0,16,'DIFFREM')
sheet3.write(0,17,'DIFFPHYS')
sheet3.write(0,18,'DIFFEYE')
sheet3.write(0,19,'DIFFHEAR')
sheet3.write(0,20,'HHWT')
sheet3.write(0,21,'PERWT')
#pumas=[pumas[1]]
for puma in pumas:
  print('\n',puma)
  for race in range(1,8):
    print('\nRace - ',race)
    ct+=1
    sheet2.write(ct,0,puma)
    sheet2.write(ct,1,race)
    #print('\nRACE = ',race)
    puma_1_df=df1
    puma_1_df=puma_1_df[(puma_1_df['PUMA']==puma)&(puma_1_df['RACE']==race)]
    #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
    feature_cols=list(puma_1_df.columns)
    #print(feature_cols)
    feature_cols.remove('INCTOT')
    feature_cols.remove('PUMA')
    feature_cols.remove('INDNAICS')
    feature_cols.remove('Unnamed: 0')
    puma_1_df=prep_hist2(puma_1_df,feature_cols)
    #print(puma_1_df['COUNT'])
    try:
      puma_1_df=adjmerge_counts_df(puma_1_df,k)
      puma_1_df=count_scale_df(puma_1_df)
    except:
      continue
    feature_cols.remove('INCTOT_DECILE')
    #print(puma_1_df.columns)
    X=puma_1_df[feature_cols]
    y=puma_1_df['INCTOT_DECILE']
    #print('\n',feature_cols)
    np.random.seed(0)
    try:
      X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    except:
      sheet2.write(ct,2,'Beep')
      #print('Not enough individuals')
      continue
    #print(y_test)
    #np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0)#, max_depth=2)
    dt_1=dt_1.fit(X_train,y_train)
    y_pred=dt_1.predict(X_test)
    sheet2.write(ct,2,metrics.accuracy_score(y_test,y_pred)*100)
    #print("Accuracy: ",metrics.accuracy_score(y_test,y_pred)*100)
    sheet2.write(ct,3,1+dt_1.tree_.max_depth)
    feature_importances = pd.DataFrame(dt_1.feature_importances_,index = X_train.columns,columns=['importance'])#.sort_values('importance', ascending=False)
    print(feature_importances)
    ctf=ct
    rown=2
    sheet3.write(ctf,0,puma)
    sheet3.write(ctf,1,race)
    for row in feature_importances.index:
      sheet3.write(ctf,rown,feature_importances['importance'][row])
      rown+=1
    sheet2.write(ct,10,len(feature_importances[feature_importances['importance']>0]))
    #accuracy for blacks
    #y_black_test=y_test[X_test['RACE']==2]
    #print(len(y_black_test))
    #X_black_test=X_test[X_test['RACE']==2]
    #print(len(X_black_test))
    #y_black_pred=dt_1.predict(X_black_test)
    #print('%age of Black People:',len(puma_1_df[(puma_1_df['RACE']==2)].index)/len(puma_1_df)*100)# & (puma_1_df['RACE']==2)]
    #print('%age of Black Women:',len(puma_1_df[(puma_1_df['RACE']==2)& (puma_1_df['SEX']==2)].index)/len(puma_1_df)*100)
    #print('%age of Non-White People:',len(puma_1_df[puma_1_df['RACE']!=1].index)/len(puma_1_df)*100)
    #print('%age of Women:',len(puma_1_df[puma_1_df['SEX']==2].index)/len(puma_1_df)*100)
    #print('%age of People > 60 yrs:',len(puma_1_df[(puma_1_df['AGE']>60)].index)/len(puma_1_df)*100)
    #print("Percentage = ",(len(puma_1_df_black.index)/len(puma_1_df)*100))
    max_d=dt_1.tree_.max_depth
    acc_list=[]
    for i in range(1,max_d+1):
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
      dt_1=dt_1.fit(X_train,y_train)
      y_pred=dt_1.predict(X_test)
      acc_list.append(metrics.accuracy_score(y_test,y_pred)*100)
    try:
      sheet2.write(ct,4,2+acc_list.index(max(acc_list)))
      #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
      #print("Max Accuracy:",max(acc_list))
      sheet2.write(ct,5,max(acc_list))
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=9)
      dt_1=dt_1.fit(X_train,y_train)
      #print(1+dt_1.tree_.max_depth)
      y_pred=dt_1.predict(X_test)
      sheet2.write(ct,6,(metrics.accuracy_score(y_test,y_pred)*100))
      #print('Accuracy for Max Height 10:',(metrics.accuracy_score(y_test,y_pred)*100))
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=19)
      dt_1=dt_1.fit(X_train,y_train)
      #print(1+dt_1.tree_.max_depth)
      y_pred=dt_1.predict(X_test)
      sheet2.write(ct,7,(metrics.accuracy_score(y_test,y_pred)*100))
      #print('Accuracy for Max Height 20:',(metrics.accuracy_score(y_test,y_pred)*100))
      sheet2.write(ct,8,round((len(df1[(df1['PUMA']==puma)&(df1['RACE']==race)])/len(df1[df1['PUMA']==puma]))*100,3))
    except:
      sheet2.write(ct,4,'Uh oh')
      continue
    #print("Accuracy for black rows is ",metrics.accuracy_score(y_black_test,y_black_pred)*100)
    #tree.plot_tree(dt_1);
    
df1=pd.read_csv("tx_acs_2019.csv")
pumas=list(df1['PUMA'].unique())
ct=0
k=10
sheet2=wbb.add_sheet('TX - AM 10')
sheet3=wbb.add_sheet('TX Feat Imp - AM 10')
sheet2.write(0, 0, 'PUMAs')
sheet2.write(0, 1, 'Race')
sheet2.write(0, 2, 'Accuracy')
sheet2.write(0, 3, 'Height')
sheet2.write(0, 4, 'Opt Ht')
sheet2.write(0, 5, 'Opt Acc')
sheet2.write(0, 6, 'Max Ht 10')
sheet2.write(0, 7, 'Max Ht 20')
sheet2.write(0, 8, 'Pop %')
sheet3.write(0,0,'PUMAs')
sheet3.write(0,1,'Race')
sheet3.write(0,2,'GQ')
sheet3.write(0,3,'OWNERSHP')
sheet3.write(0,4,'FAMSIZE')
sheet3.write(0,5,'NCHILD')
sheet3.write(0,6,'AGE')
sheet3.write(0,7,'SEX')
sheet3.write(0,8,'MARST')
sheet3.write(0,9,'RACE')
sheet3.write(0,10,'HISPAN')
sheet3.write(0,11,'EDUC')
sheet3.write(0,12,'SECTOR')
sheet3.write(0,13,'POVERTY')
sheet3.write(0,14,'DENSITY')
sheet3.write(0,15,'VETDISAB')
sheet3.write(0,16,'DIFFREM')
sheet3.write(0,17,'DIFFPHYS')
sheet3.write(0,18,'DIFFEYE')
sheet3.write(0,19,'DIFFHEAR')
sheet3.write(0,20,'HHWT')
sheet3.write(0,21,'PERWT')
#pumas=[pumas[1]]
for puma in pumas:
  print('\n',puma)
  for race in range(1,8):
    print('\nRace - ',race)
    ct+=1
    sheet2.write(ct,0,puma)
    sheet2.write(ct,1,race)
    #print('\nRACE = ',race)
    puma_1_df=df1
    puma_1_df=puma_1_df[(puma_1_df['PUMA']==puma)&(puma_1_df['RACE']==race)]
    #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
    feature_cols=list(puma_1_df.columns)
    #print(feature_cols)
    feature_cols.remove('INCTOT')
    feature_cols.remove('PUMA')
    feature_cols.remove('INDNAICS')
    feature_cols.remove('Unnamed: 0')
    puma_1_df=prep_hist2(puma_1_df,feature_cols)
    #print(puma_1_df['COUNT'])
    try:
      puma_1_df=adjmerge_counts_df(puma_1_df,k)
      puma_1_df=count_scale_df(puma_1_df)
    except:
      continue
    feature_cols.remove('INCTOT_DECILE')
    #print(puma_1_df.columns)
    X=puma_1_df[feature_cols]
    y=puma_1_df['INCTOT_DECILE']
    #print('\n',feature_cols)
    np.random.seed(0)
    try:
      X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    except:
      sheet2.write(ct,2,'Beep')
      #print('Not enough individuals')
      continue
    #print(y_test)
    #np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0)#, max_depth=2)
    dt_1=dt_1.fit(X_train,y_train)
    y_pred=dt_1.predict(X_test)
    sheet2.write(ct,2,metrics.accuracy_score(y_test,y_pred)*100)
    #print("Accuracy: ",metrics.accuracy_score(y_test,y_pred)*100)
    sheet2.write(ct,3,1+dt_1.tree_.max_depth)
    feature_importances = pd.DataFrame(dt_1.feature_importances_,index = X_train.columns,columns=['importance'])#.sort_values('importance', ascending=False)
    print(feature_importances)
    ctf=ct
    rown=2
    sheet3.write(ctf,0,puma)
    sheet3.write(ctf,1,race)
    for row in feature_importances.index:
      sheet3.write(ctf,rown,feature_importances['importance'][row])
      rown+=1
    sheet2.write(ct,10,len(feature_importances[feature_importances['importance']>0]))
    #accuracy for blacks
    #y_black_test=y_test[X_test['RACE']==2]
    #print(len(y_black_test))
    #X_black_test=X_test[X_test['RACE']==2]
    #print(len(X_black_test))
    #y_black_pred=dt_1.predict(X_black_test)
    #print('%age of Black People:',len(puma_1_df[(puma_1_df['RACE']==2)].index)/len(puma_1_df)*100)# & (puma_1_df['RACE']==2)]
    #print('%age of Black Women:',len(puma_1_df[(puma_1_df['RACE']==2)& (puma_1_df['SEX']==2)].index)/len(puma_1_df)*100)
    #print('%age of Non-White People:',len(puma_1_df[puma_1_df['RACE']!=1].index)/len(puma_1_df)*100)
    #print('%age of Women:',len(puma_1_df[puma_1_df['SEX']==2].index)/len(puma_1_df)*100)
    #print('%age of People > 60 yrs:',len(puma_1_df[(puma_1_df['AGE']>60)].index)/len(puma_1_df)*100)
    #print("Percentage = ",(len(puma_1_df_black.index)/len(puma_1_df)*100))
    max_d=dt_1.tree_.max_depth
    acc_list=[]
    for i in range(1,max_d+1):
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
      dt_1=dt_1.fit(X_train,y_train)
      y_pred=dt_1.predict(X_test)
      acc_list.append(metrics.accuracy_score(y_test,y_pred)*100)
    try:
      sheet2.write(ct,4,2+acc_list.index(max(acc_list)))
      #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
      #print("Max Accuracy:",max(acc_list))
      sheet2.write(ct,5,max(acc_list))
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=9)
      dt_1=dt_1.fit(X_train,y_train)
      #print(1+dt_1.tree_.max_depth)
      y_pred=dt_1.predict(X_test)
      sheet2.write(ct,6,(metrics.accuracy_score(y_test,y_pred)*100))
      #print('Accuracy for Max Height 10:',(metrics.accuracy_score(y_test,y_pred)*100))
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=19)
      dt_1=dt_1.fit(X_train,y_train)
      #print(1+dt_1.tree_.max_depth)
      y_pred=dt_1.predict(X_test)
      sheet2.write(ct,7,(metrics.accuracy_score(y_test,y_pred)*100))
      #print('Accuracy for Max Height 20:',(metrics.accuracy_score(y_test,y_pred)*100))
      sheet2.write(ct,8,round((len(df1[(df1['PUMA']==puma)&(df1['RACE']==race)])/len(df1[df1['PUMA']==puma]))*100,3))
    except:
      sheet2.write(ct,4,'Uh oh')
      continue
    #print("Accuracy for black rows is ",metrics.accuracy_score(y_black_test,y_black_pred)*100)
    #tree.plot_tree(dt_1);

df1=pd.read_csv("ma_acs_2019.csv")
pumas=list(df1['PUMA'].unique())
ct=0
k=6
sheet2=wbb.add_sheet('MA - AM 6')
sheet3=wbb.add_sheet('MA Feat Imp - AM 6')
sheet2.write(0, 0, 'PUMAs')
sheet2.write(0, 1, 'Race')
sheet2.write(0, 2, 'Accuracy')
sheet2.write(0, 3, 'Height')
sheet2.write(0, 4, 'Opt Ht')
sheet2.write(0, 5, 'Opt Acc')
sheet2.write(0, 6, 'Max Ht 10')
sheet2.write(0, 7, 'Max Ht 20')
sheet2.write(0, 8, 'Pop %')
sheet3.write(0,0,'PUMAs')
sheet3.write(0,1,'Race')
sheet3.write(0,2,'GQ')
sheet3.write(0,3,'OWNERSHP')
sheet3.write(0,4,'FAMSIZE')
sheet3.write(0,5,'NCHILD')
sheet3.write(0,6,'AGE')
sheet3.write(0,7,'SEX')
sheet3.write(0,8,'MARST')
sheet3.write(0,9,'RACE')
sheet3.write(0,10,'HISPAN')
sheet3.write(0,11,'EDUC')
sheet3.write(0,12,'SECTOR')
sheet3.write(0,13,'POVERTY')
sheet3.write(0,14,'DENSITY')
sheet3.write(0,15,'VETDISAB')
sheet3.write(0,16,'DIFFREM')
sheet3.write(0,17,'DIFFPHYS')
sheet3.write(0,18,'DIFFEYE')
sheet3.write(0,19,'DIFFHEAR')
sheet3.write(0,20,'HHWT')
sheet3.write(0,21,'PERWT')
#pumas=[pumas[1]]
for puma in pumas:
  print('\n',puma)
  for race in range(1,8):
    print('\nRace - ',race)
    ct+=1
    sheet2.write(ct,0,puma)
    sheet2.write(ct,1,race)
    #print('\nRACE = ',race)
    puma_1_df=df1
    puma_1_df=puma_1_df[(puma_1_df['PUMA']==puma)&(puma_1_df['RACE']==race)]
    #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
    feature_cols=list(puma_1_df.columns)
    #print(feature_cols)
    feature_cols.remove('INCTOT')
    feature_cols.remove('PUMA')
    feature_cols.remove('INDNAICS')
    feature_cols.remove('Unnamed: 0')
    puma_1_df=prep_hist2(puma_1_df,feature_cols)
    #print(puma_1_df['COUNT'])
    try:
      puma_1_df=adjmerge_counts_df(puma_1_df,k)
      puma_1_df=count_scale_df(puma_1_df)
    except:
      continue
    feature_cols.remove('INCTOT_DECILE')
    #print(puma_1_df.columns)
    X=puma_1_df[feature_cols]
    y=puma_1_df['INCTOT_DECILE']
    #print('\n',feature_cols)
    np.random.seed(0)
    try:
      X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    except:
      sheet2.write(ct,2,'Beep')
      #print('Not enough individuals')
      continue
    #print(y_test)
    #np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0)#, max_depth=2)
    dt_1=dt_1.fit(X_train,y_train)
    y_pred=dt_1.predict(X_test)
    sheet2.write(ct,2,metrics.accuracy_score(y_test,y_pred)*100)
    #print("Accuracy: ",metrics.accuracy_score(y_test,y_pred)*100)
    sheet2.write(ct,3,1+dt_1.tree_.max_depth)
    feature_importances = pd.DataFrame(dt_1.feature_importances_,index = X_train.columns,columns=['importance'])#.sort_values('importance', ascending=False)
    print(feature_importances)
    ctf=ct
    rown=2
    sheet3.write(ctf,0,puma)
    sheet3.write(ctf,1,race)
    for row in feature_importances.index:
      sheet3.write(ctf,rown,feature_importances['importance'][row])
      rown+=1
    sheet2.write(ct,10,len(feature_importances[feature_importances['importance']>0]))
    #accuracy for blacks
    #y_black_test=y_test[X_test['RACE']==2]
    #print(len(y_black_test))
    #X_black_test=X_test[X_test['RACE']==2]
    #print(len(X_black_test))
    #y_black_pred=dt_1.predict(X_black_test)
    #print('%age of Black People:',len(puma_1_df[(puma_1_df['RACE']==2)].index)/len(puma_1_df)*100)# & (puma_1_df['RACE']==2)]
    #print('%age of Black Women:',len(puma_1_df[(puma_1_df['RACE']==2)& (puma_1_df['SEX']==2)].index)/len(puma_1_df)*100)
    #print('%age of Non-White People:',len(puma_1_df[puma_1_df['RACE']!=1].index)/len(puma_1_df)*100)
    #print('%age of Women:',len(puma_1_df[puma_1_df['SEX']==2].index)/len(puma_1_df)*100)
    #print('%age of People > 60 yrs:',len(puma_1_df[(puma_1_df['AGE']>60)].index)/len(puma_1_df)*100)
    #print("Percentage = ",(len(puma_1_df_black.index)/len(puma_1_df)*100))
    max_d=dt_1.tree_.max_depth
    acc_list=[]
    for i in range(1,max_d+1):
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
      dt_1=dt_1.fit(X_train,y_train)
      y_pred=dt_1.predict(X_test)
      acc_list.append(metrics.accuracy_score(y_test,y_pred)*100)
    try:
      sheet2.write(ct,4,2+acc_list.index(max(acc_list)))
      #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
      #print("Max Accuracy:",max(acc_list))
      sheet2.write(ct,5,max(acc_list))
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=9)
      dt_1=dt_1.fit(X_train,y_train)
      #print(1+dt_1.tree_.max_depth)
      y_pred=dt_1.predict(X_test)
      sheet2.write(ct,6,(metrics.accuracy_score(y_test,y_pred)*100))
      #print('Accuracy for Max Height 10:',(metrics.accuracy_score(y_test,y_pred)*100))
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=19)
      dt_1=dt_1.fit(X_train,y_train)
      #print(1+dt_1.tree_.max_depth)
      y_pred=dt_1.predict(X_test)
      sheet2.write(ct,7,(metrics.accuracy_score(y_test,y_pred)*100))
      #print('Accuracy for Max Height 20:',(metrics.accuracy_score(y_test,y_pred)*100))
      sheet2.write(ct,8,round((len(df1[(df1['PUMA']==puma)&(df1['RACE']==race)])/len(df1[df1['PUMA']==puma]))*100,3))
    except:
      sheet2.write(ct,4,'Uh oh')
      continue
    #print("Accuracy for black rows is ",metrics.accuracy_score(y_black_test,y_black_pred)*100)
    #tree.plot_tree(dt_1);

df1=pd.read_csv("ma_acs_2019.csv")
pumas=list(df1['PUMA'].unique())
ct=0
k=10
sheet2=wbb.add_sheet('MA - AM 10')
sheet3=wbb.add_sheet('MA Feat Imp - AM 10')
sheet2.write(0, 0, 'PUMAs')
sheet2.write(0, 1, 'Race')
sheet2.write(0, 2, 'Accuracy')
sheet2.write(0, 3, 'Height')
sheet2.write(0, 4, 'Opt Ht')
sheet2.write(0, 5, 'Opt Acc')
sheet2.write(0, 6, 'Max Ht 10')
sheet2.write(0, 7, 'Max Ht 20')
sheet2.write(0, 8, 'Pop %')
sheet3.write(0,0,'PUMAs')
sheet3.write(0,1,'Race')
sheet3.write(0,2,'GQ')
sheet3.write(0,3,'OWNERSHP')
sheet3.write(0,4,'FAMSIZE')
sheet3.write(0,5,'NCHILD')
sheet3.write(0,6,'AGE')
sheet3.write(0,7,'SEX')
sheet3.write(0,8,'MARST')
sheet3.write(0,9,'RACE')
sheet3.write(0,10,'HISPAN')
sheet3.write(0,11,'EDUC')
sheet3.write(0,12,'SECTOR')
sheet3.write(0,13,'POVERTY')
sheet3.write(0,14,'DENSITY')
sheet3.write(0,15,'VETDISAB')
sheet3.write(0,16,'DIFFREM')
sheet3.write(0,17,'DIFFPHYS')
sheet3.write(0,18,'DIFFEYE')
sheet3.write(0,19,'DIFFHEAR')
sheet3.write(0,20,'HHWT')
sheet3.write(0,21,'PERWT')
#pumas=[pumas[1]]
for puma in pumas:
  print('\n',puma)
  for race in range(1,8):
    print('\nRace - ',race)
    ct+=1
    sheet2.write(ct,0,puma)
    sheet2.write(ct,1,race)
    #print('\nRACE = ',race)
    puma_1_df=df1
    puma_1_df=puma_1_df[(puma_1_df['PUMA']==puma)&(puma_1_df['RACE']==race)]
    #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
    feature_cols=list(puma_1_df.columns)
    #print(feature_cols)
    feature_cols.remove('INCTOT')
    feature_cols.remove('PUMA')
    feature_cols.remove('INDNAICS')
    feature_cols.remove('Unnamed: 0')
    puma_1_df=prep_hist2(puma_1_df,feature_cols)
    #print(puma_1_df['COUNT'])
    try:
      puma_1_df=adjmerge_counts_df(puma_1_df,k)
      puma_1_df=count_scale_df(puma_1_df)
    except:
      continue
    feature_cols.remove('INCTOT_DECILE')
    #print(puma_1_df.columns)
    X=puma_1_df[feature_cols]
    y=puma_1_df['INCTOT_DECILE']
    #print('\n',feature_cols)
    np.random.seed(0)
    try:
      X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    except:
      sheet2.write(ct,2,'Beep')
      #print('Not enough individuals')
      continue
    #print(y_test)
    #np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0)#, max_depth=2)
    dt_1=dt_1.fit(X_train,y_train)
    y_pred=dt_1.predict(X_test)
    sheet2.write(ct,2,metrics.accuracy_score(y_test,y_pred)*100)
    #print("Accuracy: ",metrics.accuracy_score(y_test,y_pred)*100)
    sheet2.write(ct,3,1+dt_1.tree_.max_depth)
    feature_importances = pd.DataFrame(dt_1.feature_importances_,index = X_train.columns,columns=['importance'])#.sort_values('importance', ascending=False)
    print(feature_importances)
    ctf=ct
    rown=2
    sheet3.write(ctf,0,puma)
    sheet3.write(ctf,1,race)
    for row in feature_importances.index:
      sheet3.write(ctf,rown,feature_importances['importance'][row])
      rown+=1
    sheet2.write(ct,10,len(feature_importances[feature_importances['importance']>0]))
    #accuracy for blacks
    #y_black_test=y_test[X_test['RACE']==2]
    #print(len(y_black_test))
    #X_black_test=X_test[X_test['RACE']==2]
    #print(len(X_black_test))
    #y_black_pred=dt_1.predict(X_black_test)
    #print('%age of Black People:',len(puma_1_df[(puma_1_df['RACE']==2)].index)/len(puma_1_df)*100)# & (puma_1_df['RACE']==2)]
    #print('%age of Black Women:',len(puma_1_df[(puma_1_df['RACE']==2)& (puma_1_df['SEX']==2)].index)/len(puma_1_df)*100)
    #print('%age of Non-White People:',len(puma_1_df[puma_1_df['RACE']!=1].index)/len(puma_1_df)*100)
    #print('%age of Women:',len(puma_1_df[puma_1_df['SEX']==2].index)/len(puma_1_df)*100)
    #print('%age of People > 60 yrs:',len(puma_1_df[(puma_1_df['AGE']>60)].index)/len(puma_1_df)*100)
    #print("Percentage = ",(len(puma_1_df_black.index)/len(puma_1_df)*100))
    max_d=dt_1.tree_.max_depth
    acc_list=[]
    for i in range(1,max_d+1):
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
      dt_1=dt_1.fit(X_train,y_train)
      y_pred=dt_1.predict(X_test)
      acc_list.append(metrics.accuracy_score(y_test,y_pred)*100)
    try:
      sheet2.write(ct,4,2+acc_list.index(max(acc_list)))
      #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
      #print("Max Accuracy:",max(acc_list))
      sheet2.write(ct,5,max(acc_list))
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=9)
      dt_1=dt_1.fit(X_train,y_train)
      #print(1+dt_1.tree_.max_depth)
      y_pred=dt_1.predict(X_test)
      sheet2.write(ct,6,(metrics.accuracy_score(y_test,y_pred)*100))
      #print('Accuracy for Max Height 10:',(metrics.accuracy_score(y_test,y_pred)*100))
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=19)
      dt_1=dt_1.fit(X_train,y_train)
      #print(1+dt_1.tree_.max_depth)
      y_pred=dt_1.predict(X_test)
      sheet2.write(ct,7,(metrics.accuracy_score(y_test,y_pred)*100))
      #print('Accuracy for Max Height 20:',(metrics.accuracy_score(y_test,y_pred)*100))
      sheet2.write(ct,8,round((len(df1[(df1['PUMA']==puma)&(df1['RACE']==race)])/len(df1[df1['PUMA']==puma]))*100,3))
    except:
      sheet2.write(ct,4,'Uh oh')
      continue
    #print("Accuracy for black rows is ",metrics.accuracy_score(y_black_test,y_black_pred)*100)
    #tree.plot_tree(dt_1);

df1=pd.read_csv("outlier_acs_2019.csv")
pumas=list(df1['PUMA'].unique())
ct=0
k=6
sheet2=wbb.add_sheet('Outlier - AM 6')
sheet3=wbb.add_sheet('Outlier Feat Imp - AM 6')
sheet2.write(0, 0, 'PUMAs')
sheet2.write(0, 1, 'Race')
sheet2.write(0, 2, 'Accuracy')
sheet2.write(0, 3, 'Height')
sheet2.write(0, 4, 'Opt Ht')
sheet2.write(0, 5, 'Opt Acc')
sheet2.write(0, 6, 'Max Ht 10')
sheet2.write(0, 7, 'Max Ht 20')
sheet2.write(0, 8, 'Pop %')
sheet3.write(0,0,'PUMAs')
sheet3.write(0,1,'Race')
sheet3.write(0,2,'GQ')
sheet3.write(0,3,'OWNERSHP')
sheet3.write(0,4,'FAMSIZE')
sheet3.write(0,5,'NCHILD')
sheet3.write(0,6,'AGE')
sheet3.write(0,7,'SEX')
sheet3.write(0,8,'MARST')
sheet3.write(0,9,'RACE')
sheet3.write(0,10,'HISPAN')
sheet3.write(0,11,'EDUC')
sheet3.write(0,12,'SECTOR')
sheet3.write(0,13,'POVERTY')
sheet3.write(0,14,'DENSITY')
sheet3.write(0,15,'VETDISAB')
sheet3.write(0,16,'DIFFREM')
sheet3.write(0,17,'DIFFPHYS')
sheet3.write(0,18,'DIFFEYE')
sheet3.write(0,19,'DIFFHEAR')
sheet3.write(0,20,'HHWT')
sheet3.write(0,21,'PERWT')
#pumas=[pumas[1]]
for puma in pumas:
  print('\n',puma)
  for race in range(1,8):
    print('\nRace - ',race)
    ct+=1
    sheet2.write(ct,0,puma)
    sheet2.write(ct,1,race)
    #print('\nRACE = ',race)
    puma_1_df=df1
    puma_1_df=puma_1_df[(puma_1_df['PUMA']==puma)&(puma_1_df['RACE']==race)]
    #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
    feature_cols=list(puma_1_df.columns)
    #print(feature_cols)
    feature_cols.remove('INCTOT')
    feature_cols.remove('PUMA')
    feature_cols.remove('INDNAICS')
    feature_cols.remove('Unnamed: 0')
    puma_1_df=prep_hist2(puma_1_df,feature_cols)
    #print(puma_1_df['COUNT'])
    try:
      puma_1_df=adjmerge_counts_df(puma_1_df,k)
      puma_1_df=count_scale_df(puma_1_df)
    except:
      continue
    feature_cols.remove('INCTOT_DECILE')
    #print(puma_1_df.columns)
    X=puma_1_df[feature_cols]
    y=puma_1_df['INCTOT_DECILE']
    #print('\n',feature_cols)
    np.random.seed(0)
    try:
      X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    except:
      sheet2.write(ct,2,'Beep')
      #print('Not enough individuals')
      continue
    #print(y_test)
    #np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0)#, max_depth=2)
    dt_1=dt_1.fit(X_train,y_train)
    y_pred=dt_1.predict(X_test)
    sheet2.write(ct,2,metrics.accuracy_score(y_test,y_pred)*100)
    #print("Accuracy: ",metrics.accuracy_score(y_test,y_pred)*100)
    sheet2.write(ct,3,1+dt_1.tree_.max_depth)
    feature_importances = pd.DataFrame(dt_1.feature_importances_,index = X_train.columns,columns=['importance'])#.sort_values('importance', ascending=False)
    print(feature_importances)
    ctf=ct
    rown=2
    sheet3.write(ctf,0,puma)
    sheet3.write(ctf,1,race)
    for row in feature_importances.index:
      sheet3.write(ctf,rown,feature_importances['importance'][row])
      rown+=1
    sheet2.write(ct,10,len(feature_importances[feature_importances['importance']>0]))
    #accuracy for blacks
    #y_black_test=y_test[X_test['RACE']==2]
    #print(len(y_black_test))
    #X_black_test=X_test[X_test['RACE']==2]
    #print(len(X_black_test))
    #y_black_pred=dt_1.predict(X_black_test)
    #print('%age of Black People:',len(puma_1_df[(puma_1_df['RACE']==2)].index)/len(puma_1_df)*100)# & (puma_1_df['RACE']==2)]
    #print('%age of Black Women:',len(puma_1_df[(puma_1_df['RACE']==2)& (puma_1_df['SEX']==2)].index)/len(puma_1_df)*100)
    #print('%age of Non-White People:',len(puma_1_df[puma_1_df['RACE']!=1].index)/len(puma_1_df)*100)
    #print('%age of Women:',len(puma_1_df[puma_1_df['SEX']==2].index)/len(puma_1_df)*100)
    #print('%age of People > 60 yrs:',len(puma_1_df[(puma_1_df['AGE']>60)].index)/len(puma_1_df)*100)
    #print("Percentage = ",(len(puma_1_df_black.index)/len(puma_1_df)*100))
    max_d=dt_1.tree_.max_depth
    acc_list=[]
    for i in range(1,max_d+1):
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
      dt_1=dt_1.fit(X_train,y_train)
      y_pred=dt_1.predict(X_test)
      acc_list.append(metrics.accuracy_score(y_test,y_pred)*100)
    try:
      sheet2.write(ct,4,2+acc_list.index(max(acc_list)))
      #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
      #print("Max Accuracy:",max(acc_list))
      sheet2.write(ct,5,max(acc_list))
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=9)
      dt_1=dt_1.fit(X_train,y_train)
      #print(1+dt_1.tree_.max_depth)
      y_pred=dt_1.predict(X_test)
      sheet2.write(ct,6,(metrics.accuracy_score(y_test,y_pred)*100))
      #print('Accuracy for Max Height 10:',(metrics.accuracy_score(y_test,y_pred)*100))
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=19)
      dt_1=dt_1.fit(X_train,y_train)
      #print(1+dt_1.tree_.max_depth)
      y_pred=dt_1.predict(X_test)
      sheet2.write(ct,7,(metrics.accuracy_score(y_test,y_pred)*100))
      #print('Accuracy for Max Height 20:',(metrics.accuracy_score(y_test,y_pred)*100))
      sheet2.write(ct,8,round((len(df1[(df1['PUMA']==puma)&(df1['RACE']==race)])/len(df1[df1['PUMA']==puma]))*100,3))
    except:
      sheet2.write(ct,4,'Uh oh')
      continue
    #print("Accuracy for black rows is ",metrics.accuracy_score(y_black_test,y_black_pred)*100)
    #tree.plot_tree(dt_1);

df1=pd.read_csv("outlier_acs_2019.csv")
pumas=list(df1['PUMA'].unique())
ct=0
k=10
sheet2=wbb.add_sheet('Outlier - AM 10')
sheet3=wbb.add_sheet('Outlier Feat Imp - AM 10')
sheet2.write(0, 0, 'PUMAs')
sheet2.write(0, 1, 'Race')
sheet2.write(0, 2, 'Accuracy')
sheet2.write(0, 3, 'Height')
sheet2.write(0, 4, 'Opt Ht')
sheet2.write(0, 5, 'Opt Acc')
sheet2.write(0, 6, 'Max Ht 10')
sheet2.write(0, 7, 'Max Ht 20')
sheet2.write(0, 8, 'Pop %')
sheet3.write(0,0,'PUMAs')
sheet3.write(0,1,'Race')
sheet3.write(0,2,'GQ')
sheet3.write(0,3,'OWNERSHP')
sheet3.write(0,4,'FAMSIZE')
sheet3.write(0,5,'NCHILD')
sheet3.write(0,6,'AGE')
sheet3.write(0,7,'SEX')
sheet3.write(0,8,'MARST')
sheet3.write(0,9,'RACE')
sheet3.write(0,10,'HISPAN')
sheet3.write(0,11,'EDUC')
sheet3.write(0,12,'SECTOR')
sheet3.write(0,13,'POVERTY')
sheet3.write(0,14,'DENSITY')
sheet3.write(0,15,'VETDISAB')
sheet3.write(0,16,'DIFFREM')
sheet3.write(0,17,'DIFFPHYS')
sheet3.write(0,18,'DIFFEYE')
sheet3.write(0,19,'DIFFHEAR')
sheet3.write(0,20,'HHWT')
sheet3.write(0,21,'PERWT')
#pumas=[pumas[1]]
for puma in pumas:
  print('\n',puma)
  for race in range(1,8):
    print('\nRace - ',race)
    ct+=1
    sheet2.write(ct,0,puma)
    sheet2.write(ct,1,race)
    #print('\nRACE = ',race)
    puma_1_df=df1
    puma_1_df=puma_1_df[(puma_1_df['PUMA']==puma)&(puma_1_df['RACE']==race)]
    #feature_cols=['OWNERSHP','FAMSIZE','NCHILD','AGE','SEX','RACE','EDUC','SECTOR']
    feature_cols=list(puma_1_df.columns)
    #print(feature_cols)
    feature_cols.remove('INCTOT')
    feature_cols.remove('PUMA')
    feature_cols.remove('INDNAICS')
    feature_cols.remove('Unnamed: 0')
    puma_1_df=prep_hist2(puma_1_df,feature_cols)
    #print(puma_1_df['COUNT'])
    try:
      puma_1_df=adjmerge_counts_df(puma_1_df,k)
      puma_1_df=count_scale_df(puma_1_df)
    except:
      continue
    feature_cols.remove('INCTOT_DECILE')
    #print(puma_1_df.columns)
    X=puma_1_df[feature_cols]
    y=puma_1_df['INCTOT_DECILE']
    #print('\n',feature_cols)
    np.random.seed(0)
    try:
      X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    except:
      sheet2.write(ct,2,'Beep')
      #print('Not enough individuals')
      continue
    #print(y_test)
    #np.random.seed(0)
    dt_1=DecisionTreeClassifier(random_state=0)#, max_depth=2)
    dt_1=dt_1.fit(X_train,y_train)
    y_pred=dt_1.predict(X_test)
    sheet2.write(ct,2,metrics.accuracy_score(y_test,y_pred)*100)
    #print("Accuracy: ",metrics.accuracy_score(y_test,y_pred)*100)
    sheet2.write(ct,3,1+dt_1.tree_.max_depth)
    feature_importances = pd.DataFrame(dt_1.feature_importances_,index = X_train.columns,columns=['importance'])#.sort_values('importance', ascending=False)
    print(feature_importances)
    ctf=ct
    rown=2
    sheet3.write(ctf,0,puma)
    sheet3.write(ctf,1,race)
    for row in feature_importances.index:
      sheet3.write(ctf,rown,feature_importances['importance'][row])
      rown+=1
    sheet2.write(ct,10,len(feature_importances[feature_importances['importance']>0]))
    #accuracy for blacks
    #y_black_test=y_test[X_test['RACE']==2]
    #print(len(y_black_test))
    #X_black_test=X_test[X_test['RACE']==2]
    #print(len(X_black_test))
    #y_black_pred=dt_1.predict(X_black_test)
    #print('%age of Black People:',len(puma_1_df[(puma_1_df['RACE']==2)].index)/len(puma_1_df)*100)# & (puma_1_df['RACE']==2)]
    #print('%age of Black Women:',len(puma_1_df[(puma_1_df['RACE']==2)& (puma_1_df['SEX']==2)].index)/len(puma_1_df)*100)
    #print('%age of Non-White People:',len(puma_1_df[puma_1_df['RACE']!=1].index)/len(puma_1_df)*100)
    #print('%age of Women:',len(puma_1_df[puma_1_df['SEX']==2].index)/len(puma_1_df)*100)
    #print('%age of People > 60 yrs:',len(puma_1_df[(puma_1_df['AGE']>60)].index)/len(puma_1_df)*100)
    #print("Percentage = ",(len(puma_1_df_black.index)/len(puma_1_df)*100))
    max_d=dt_1.tree_.max_depth
    acc_list=[]
    for i in range(1,max_d+1):
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
      dt_1=dt_1.fit(X_train,y_train)
      y_pred=dt_1.predict(X_test)
      acc_list.append(metrics.accuracy_score(y_test,y_pred)*100)
    try:
      sheet2.write(ct,4,2+acc_list.index(max(acc_list)))
      #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
      #print("Max Accuracy:",max(acc_list))
      sheet2.write(ct,5,max(acc_list))
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=9)
      dt_1=dt_1.fit(X_train,y_train)
      #print(1+dt_1.tree_.max_depth)
      y_pred=dt_1.predict(X_test)
      sheet2.write(ct,6,(metrics.accuracy_score(y_test,y_pred)*100))
      #print('Accuracy for Max Height 10:',(metrics.accuracy_score(y_test,y_pred)*100))
      np.random.seed(0)
      dt_1=DecisionTreeClassifier(random_state=0, max_depth=19)
      dt_1=dt_1.fit(X_train,y_train)
      #print(1+dt_1.tree_.max_depth)
      y_pred=dt_1.predict(X_test)
      sheet2.write(ct,7,(metrics.accuracy_score(y_test,y_pred)*100))
      #print('Accuracy for Max Height 20:',(metrics.accuracy_score(y_test,y_pred)*100))
      sheet2.write(ct,8,round((len(df1[(df1['PUMA']==puma)&(df1['RACE']==race)])/len(df1[df1['PUMA']==puma]))*100,3))
    except:
      sheet2.write(ct,4,'Uh oh')
      continue
    #print("Accuracy for black rows is ",metrics.accuracy_score(y_black_test,y_black_pred)*100)
    #tree.plot_tree(dt_1);
wbb.save('wbbAMPUMA.xls')
