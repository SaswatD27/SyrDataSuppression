#prepare hist
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
  #print('hey')
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
#Get Randmerge Stats
def dedicated_dt_randmerge(dfname,shortdfname,k,wbb): #init wbb=Workbook() prior to running
  df1=pd.read_csv(dfname)#"tx_acs_2019.csv")
  pumas=list(df1['PUMA'].unique())
  ct=0
  #k=6
  sheet2=wbb.add_sheet(str(shortdfname)+' - RM '+str(k))
  sheet3=wbb.add_sheet(str(shortdfname)+' Feat Imp - RM '+str(k))
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
      #print(puma_1_df['COUNT'])
      accuracy=0
      height=0
      feature_importances_n=np.zeros(20)
      features_used=0
      #for _ in range(100):
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
      if len(puma_1_df)<2:
        continue
      accuracy=0
      height=0
      feature_importances_n=np.zeros(20)
      features_used=0
      acc_10=0
      acc_20=0
      opt_ht=0
      opt_acc=0
      for _ in range(100):
        puma_1_df=randmerge_counts_df(puma_1_df,k)
        puma_1_df=count_scale_df(puma_1_df)
        try:
          feature_cols.remove('INCTOT_DECILE')
        except:
          print(' ')
        #print(puma_1_df.columns)
        X=puma_1_df[feature_cols]
        y=puma_1_df['INCTOT_DECILE']
        #print('\n',feature_cols)
        np.random.seed(0)
        try:
          X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
        except:
          #sheet2.write(ct,2,'Beep')
          #print('Not enough individuals')
          continue
        #print(y_test)
        #np.random.seed(0)
        dt_1=DecisionTreeClassifier(random_state=0)#, max_depth=2)
        dt_1=dt_1.fit(X_train,y_train)
        y_pred=dt_1.predict(X_test)
        accuracy=(metrics.accuracy_score(y_test,y_pred)*100)
        #print("Accuracy: ",metrics.accuracy_score(y_test,y_pred)*100)
        height+=(1+dt_1.tree_.max_depth)
        print(height)
        #feature_importances=np.array(pd.DataFrame(dt_1.feature_importances_,index = X_train.columns,columns=['importance']))
        #feature_importances_n+=feature_importances['importance']
        feature_importances = pd.DataFrame(dt_1.feature_importances_,index = X_train.columns,columns=['importance'])
        feature_importances_n+=np.array(feature_importances['importance'])
        features_used+=(len(feature_importances[feature_importances['importance']>0]))
        max_d=dt_1.tree_.max_depth
        acc_list=[]
        for i in range(1,max_d+1):
          np.random.seed(0)
          dt_1=DecisionTreeClassifier(random_state=0, max_depth=i)
          dt_1=dt_1.fit(X_train,y_train)
          y_pred=dt_1.predict(X_test)
          acc_list.append(metrics.accuracy_score(y_test,y_pred)*100)
        try:
          opt_ht+=(2+acc_list.index(max(acc_list)))
          #print("Optimal Tree Height:",2+acc_list.index(max(acc_list)))
          #print("Max Accuracy:",max(acc_list))
          opt_acc+=(max(acc_list))
          np.random.seed(0)
          dt_1=DecisionTreeClassifier(random_state=0, max_depth=9)
          dt_1=dt_1.fit(X_train,y_train)
          #print(1+dt_1.tree_.max_depth)
          y_pred=dt_1.predict(X_test)
          acc_10+=((metrics.accuracy_score(y_test,y_pred)*100))
          #print('Accuracy for Max Height 10:',(metrics.accuracy_score(y_test,y_pred)*100))
          np.random.seed(0)
          dt_1=DecisionTreeClassifier(random_state=0, max_depth=19)
          dt_1=dt_1.fit(X_train,y_train)
          #print(1+dt_1.tree_.max_depth)
          y_pred=dt_1.predict(X_test)
          acc_20+=((metrics.accuracy_score(y_test,y_pred)*100))
          #print('Accuracy for Max Height 20:',(metrics.accuracy_score(y_test,y_pred)*100))
          #sheet2.write(ct,8,round((len(df1[(df1['PUMA']==puma)&(df1['RACE']==race)])/len(df1[df1['PUMA']==puma]))*100,3))
        except:
          sheet2.write(ct,4,'Uh oh')
          continue
        #print(feature_importances)
        #ctf=ct
        #rown=2
        #sheet3.write(ctf,0,puma)
        #sheet3.write(ctf,1,race)
        '''
        for row in feature_importances.index:
          sheet3.write(ctf,rown,feature_importances['importance'][row])
          rown+=1
        '''
        #feature_importances=np.array(feature_importances['importance'])
        #features_used+=(len(feature_importances[feature_importances['importance']>0]))
      feature_importances_n/=100
      features_used/=100
      accuracy/=100
      height/=100
      acc_10/=100
      acc_20/=100
      opt_ht/=100
      opt_acc/=100
      sheet2.write(ct,2,accuracy)
      sheet2.write(ct,3,height)
      sheet2.write(ct,4,opt_ht)
      sheet2.write(ct,5,opt_acc)
      sheet2.write(ct,10,features_used)
      sheet2.write(ct,8,round((len(df1[(df1['PUMA']==puma)&(df1['RACE']==race)])/len(df1[df1['PUMA']==puma]))*100,3))
      sheet2.write(ct,6,acc_10)
      sheet2.write(ct,7,acc_20)
      for rown in range(20):
        sheet3.write(ct,rown+2,feature_importances_n[rown])
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
        #print("Accuracy for black rows is ",metrics.accuracy_score(y_black_test,y_black_pred)*100)
        #tree.plot_tree(dt_1);
  wbb.save('RMforDTdedicated1.xls')
def dedicated_dt_randmerge_all():
  wbb=Workbook()
  dedicated_dt_randmerge("tx_acs_2019.csv",'TX',6,wbb)
  dedicated_dt_randmerge("tx_acs_2019.csv",'TX',10,wbb)
  dedicated_dt_randmerge("ma_acs_2019.csv",'MA',6,wbb)
  dedicated_dt_randmerge("ma_acs_2019.csv",'MA',10,wbb)
  dedicated_dt_randmerge("outlier_acs_2019.csv",'Outlier',6,wbb)
  dedicated_dt_randmerge("outlier_acs_2019.csv",'Outlier',10,wbb)

dedicated_dt_randmerge_all()
