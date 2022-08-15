#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import pipeline_dp as pdp
import math
import random

#------- Discrete Laplace Mechanism -----------
#Code Block Credits : Andrew Reed (https://mathoverflow.net/questions/213221/what-is-a-two-sided-geometric-distribution)
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
def add_dp_hist(counts,eps):
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
  #print('hi2')
  while len(cols)>0:
    print(len(cols))
    #print('hey')
    col=cols[0]
    #print(col)
    colsmerge=[col]
    colsmerge_val=[hist1[col]]
    #print(hist)
    #print(hist[col]+hist[col2])
    #while (hist[col]+hist[col2]<k) or (col2 in cols):
    while sum(colsmerge_val)<k:
      col2=random.choice([*range(0,len(hist))])
      colsmerge.append(col2)
      colsmerge_val.append(hist1[col2])
    val=sum(colsmerge_val)/len(colsmerge)
    for c in colsmerge:
      hist1[c]=val
    #hist1[col2]=val
    #hist1[col]=val
      if c in cols:
        cols=list(cols)
        cols.remove(c)
        cols=np.array(cols)
  return hist1
  '''
  hist1=list(hist)
  while len(cols)>0:
    print('hey')
    col=cols[0]
    #print(col)
    print(len(cols))
    if col == 0:
      col2=1
    else:
      col2=0
    colsmerge=[col]
    colsmerge_val=[hist1[col]]
    #print(hist)
    #print(hist[col]+hist[col2])
    #while (hist[col]+hist[col2]<k) or (col2 in cols):
    while sum(colsmerge_val)<k:
      while col2 in colsmerge:
        col2=random.choice([*range(0,len(hist))])
      colsmerge.append(col2)
      colsmerge_val.append(hist1[col2])
    #  print('lol')
    #print(col2)
    val=sum(colsmerge_val)/len(colsmerge)
    cols=list(cols)
    for c in colsmerge:
      hist1[c]=val
    #hist1[col2]=val
    #hist1[col]=val
      if c in cols:
       cols.remove(c)
    cols=np.array(cols)
  return hist1
  '''
def spread_k_anonymise(hist,k): #k anonymises a histogram with random bin merging
  hist=np.array(hist)
  #print(hist<k)
  #while any(hist<k):
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

#----- k anon adj merge ------
def spread_bins(hist,col1,col2):
  hist1=list(hist)
  hist1[col1]=(hist[col1]+hist[col2])/2
  hist1[col2]=(hist[col1]+hist[col2])/2
  return hist1

def spread_adjacent(hist,cols,k): #Merges a given bin/bucket with another bin
  hist1=list(hist)
  print('hi2')
  while len(cols)>0:
    #print(len(cols))
    #print('hey')
    col=cols[0]
    #print(col)
    colsmerge=[col]
    colsmerge_val=[hist1[col]]
    #print(hist)
    #print(hist[col]+hist[col2])
    #while (hist[col]+hist[col2]<k) or (col2 in cols):
    while sum(colsmerge_val)<k:
      if col == 0:
        col2=1
      else:
        col2=col-1
      #print(col2)
      colsmerge.append(col2)
      colsmerge_val.append(hist1[col2])
    val=sum(colsmerge_val)/len(colsmerge)
    for c in colsmerge:
      hist1[c]=val
    #hist1[col2]=val
    #hist1[col]=val
      if c in cols:
        cols=list(cols)
        cols.remove(c)
        cols=np.array(cols)
  return hist1

'''
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
def spread_adjacent(hist,cols,k): #Merges a given bin/bucket with adjacent bins
  hist1=list(hist)
  if col==len(hist)-1:
    hist1=list(spread_bins(hist,col,col-1))
    li=[col,col-1]
  elif col==0:
    hist1=list(spread_bins(hist,col,col+1))
    li=[col,col+1]
  else:
    if hist[col-1]>hist[col+1]:
      hist1=list(spread_bins(hist,col,col-1))
      li=[col,col-1]
    else:
      hist1=list(spread_bins(hist,col,col+1))
      li=[col,col+1]
  return hist1, li
'''
def adj_k_anonymise(hist,k):
  print('hi')
  hist=np.array(hist)
  i=np.where(np.array(hist)<k)
  hist=np.array(spread_adjacent(hist,i[0],k))
  return hist.astype(int)
  #hist=np.array(hist)
  #return hist.astype(int)
#-----------------------------

#----- Times DP is beaten ----
#Perhaps write it in the notebook itself
#-----------------------------
#----2.0-------
def prep_hist(df):
  mar_df_1=df.groupby(['PUMA','AGE','RACE','SEX','OWNERSHP','INCTOT_DECILE'])[['GQ']]
  hist=mar_df_1.count()
  hist.reset_index()
  hist1=hist.copy()
  for i in range(6):
    hist1=hist1.reset_index(level=0)
  print(hist1)
  hist1=hist1.rename(columns={'GQ':"COUNT"})
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
  counts_1=hist1['COUNT']
  for i in range(len(counts_1)):
    if counts_1[i]<k:
      counts_1[i]=(k//2)
  hist1['COUNT']=counts_1
  return hist1
#
def dp_counts_df(hist,eps): #Outputs redact-suppressed histogram
  hist1=hist.copy()
  counts_1=hist1['COUNT']
  try:
    dp_counts=add_dp_hist(counts_1,eps)
  except:
    dp_counts=syr.add_dp_hist(counts_1,eps)
  hist1['COUNT']=dp_counts
  return hist1
#
def randmerge_counts_df(hist,k): #Outputs redact-suppressed histogram
  hist1=hist.copy()
  counts_1=hist1['COUNT']
  try:
    dp_counts=spread_k_anonymise(counts_1,k)
  except:
    dp_counts=syr.spread_k_anonymise(counts_1,k)
  hist1['COUNT']=dp_counts
  return hist1


