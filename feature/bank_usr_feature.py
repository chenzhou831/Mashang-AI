# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 10:21:35 2018

@author: pcdalao
"""


import os
import pandas as pd
os.chdir("F:\项目\马上金融借贷预测")   #修改当前工作目录
os.getcwd()    #获取当前工作目录

#导入数据
test_list=pd.read_csv('test_list.csv')
train_target=pd.read_csv('train_target.csv')
train_bankcard_info = pd.read_csv('train_bankcard_info.csv')
test_bankcard_info=pd.read_csv('test_bankcard_info.csv')  
test_user_info=pd.read_csv('test_user_info.csv') 
train_user_info=pd.read_csv('train_user_info.csv') 

bankcard_info=pd.concat([train_bankcard_info,test_bankcard_info])#合并训练集和测试集的数据

bankcard_info.dtypes #查看变量类型

#更换变量类型
bankcard_info['id']=bankcard_info['id'].astype('category')
bankcard_info['bank_name']=bankcard_info['bank_name'].astype('category')

bankcard_info.dtypes #查看变量类型

bankcard_cnt=bankcard_info.groupby('id').count()['phone'].rename('bankcard_cnt') #卡个数
bankcard_cnt_cc=bankcard_info[(bankcard_info['card_type']=='信用卡')].groupby('id').count()['phone'].rename('bankcard_cnt_cc') #信用卡个数
bankcard_cnt_ncc=bankcard_info[(bankcard_info['card_type']=='储蓄卡')].groupby('id').count()['phone'].rename('bankcard_cnt_ncc') #信用卡个数
bankcard_card=pd.concat([pd.DataFrame(bankcard_cnt),pd.DataFrame(bankcard_cnt_cc),pd.DataFrame(bankcard_cnt_ncc)],axis=1,join='inner')

bankcard_cnt_uni=bankcard_info.groupby('id').agg({'bank_name': pd.Series.nunique,'phone':pd.Series.nunique})
bankcard_cnt_uni=bankcard_cnt_uni.rename(columns={"bank_name": "bank_cnt_bank", "phone": "phone_num"})#申请不同银行数，电话数

all_target=pd.concat([test_list,train_target])
bank=pd.merge(bankcard_info,all_target,left_on='id',right_on='id',how='inner')#看每个银行的违约数

bank_group_cnt=bank.groupby('bank_name').agg({"target":'count'}).rename(columns={"target": 'bank_group_cnt'})
bank_group_mean=bank.groupby('bank_name').agg({"target":'mean'}).rename(columns={"target": 'bank_group_mean'})

pd.merge(bank_group_cnt[(bank_group_cnt['bank_group_cnt']>1000)],bank_group_mean,left_index=True,right_index=True,how='inner').sort_values(by=['bank_group_mean','bank_group_cnt'],ascending=True).head(100)
#违约较高且样本较足的有post 信用社 abc spdb ccb
#违约较低且样本较足的有 招商银行 平安银行 中国民生银行 

#生成各银行卡数变量
bank_post=bankcard_info[bankcard_info['bank_name']=='post'].ix[:,['id','bank_name']].groupby('id').count().rename(columns={"bank_name": 'bank_post'})
bank_abc=bankcard_info[bankcard_info['bank_name']=='abc'].ix[:,['id','bank_name']].groupby('id').count().rename(columns={"bank_name": 'bank_abc'})
bank_spdb=bankcard_info[bankcard_info['bank_name']=='spdb'].ix[:,['id','bank_name']].groupby('id').count().rename(columns={"bank_name":'bank_spdb'})
bank_ccb=bankcard_info[bankcard_info['bank_name']=='ccb'].ix[:,['id','bank_name']].groupby('id').count().rename(columns={"bank_name":'bank_ccb'})
bank_zc=bankcard_info[bankcard_info['bank_name']=='招商银行'].ix[:,['id','bank_name']].groupby('id').count().rename(columns={"bank_name":'bank_zc'})
bank_pa=bankcard_info[bankcard_info['bank_name']=='平安银行'].ix[:,['id','bank_name']].groupby('id').count().rename(columns={"bank_name":'bank_pa'})
bank_ms=bankcard_info[bankcard_info['bank_name']=='中国民生银行'].ix[:,['id','bank_name']].groupby('id').count().rename(columns={"bank_name":'bank_ms'})

bankcard_info['bank_xys_index']=bankcard_info['bank_name'].astype('str').map(lambda x : x.find('信用社')>0).astype(int)

bank_xys=bankcard_info[bankcard_info['bank_xys_index']==1].ix[:,['id','bank_xys_index']].groupby('id').count().rename(columns={"bank_xys_index":'bank_xys'})

bank_bank=pd.concat([bank_post,bank_abc,bank_spdb,bank_ccb,bank_zc,bank_pa,bank_ms,bank_xys],axis=1,join='inner')
bank_final=pd.concat([bankcard_card,bankcard_cnt_uni,bank_bank],axis=1,join='inner')

bank_final #生成最后的银行卡数据

#use info-----------------------------------------
user_info=pd.concat([test_user_info,train_user_info]) #合并训练集和检验集的
all_target=pd.concat([test_list,train_target])
usr_info_target=pd.merge(user_info,all_target,left_on='id',right_on='id',how='inner')


#取年份
import numpy as np
def cut_n(x):
    if len(str(x))==0:
        return np.nan
    else:
        return (str(x).strip()[:4])
usr_info_target['birthday_cut']=usr_info_target['birthday'].map(lambda x:cut_n(x))

#只保留18岁到60岁的正常生日的人
def spe(x):
     if x not in list(map(str,list(range(1958,2001)))):
        return  '2018'
     else :
        return  x
usr_info_target['birthday_cut_clear']=usr_info_target['birthday_cut'].astype('category').map(spe).astype('int')
#sex 
usr_info_sex=usr_info_target['sex'].isnull().astype('int')
#计算贷款时年龄
usr_info_target['appl_sbm_tm']=usr_info_target['appl_sbm_tm'].map(lambda x :x.strip()[:4]).astype('int')
usr_info_year=usr_info_target['appl_sbm_tm']-usr_info_target['birthday_cut_clear']
usr_info_year=usr_info_year.rename('year')
#有无hobby
usr_info_hobby=usr_info_target['hobby'].isnull().astype('int').rename('usr_info_hobby')
#income 
usr_info_gt6000=usr_info_target.income.map(lambda x: x in ['8000元以上','6000-7999元']).astype('int').rename('usr_info_gt6000')
usr_info_lt6000=usr_info_target.income.map(lambda x: x in ['4000-5999元','2000-3999元','2000元以下']).astype('int').rename('usr_info_lt6000')
usr_info_income=usr_info_target['income'].isnull().astype('int').rename('usr_info_income')
#degree
usr_info_degree_high=usr_info_target.degree.map(lambda x: x in ['大专','本科','硕士','博士']).astype('int').rename('usr_info_degree_high')
usr_info_degree_low=usr_info_target.degree.map(lambda x: x in ['高中','中专','初中','其他']).astype('int').rename('usr_info_degree_low')
usr_info_degree=usr_info_target['degree'].isnull().astype('int').rename('usr_info_degree')
#industry
usr_info_industry=usr_info_target['industry'].isnull().astype('int').rename('usr_info_industry')
#qq_bound
usr_info_qq_bound=usr_info_target['qq_bound'].isnull().astype('int').rename('usr_info_qq_bound')
#wechat_bound
usr_info_wechat_bound=usr_info_target['wechat_bound'].isnull().astype('int')
#account_grade
usr_info_account_grade=usr_info_target['account_grade'].isnull().astype('int')
#get_dummy
usr_info_dummy=pd.get_dummies(usr_info_target.ix[:,['sex','qq_bound','wechat_bound','account_grade']])
#usr_all
usr_all=pd.concat([usr_info_sex,usr_info_year,usr_info_hobby,usr_info_gt6000,usr_info_lt6000,usr_info_income,usr_info_degree_high,
                   usr_info_degree_low,usr_info_degree,usr_info_industry,usr_info_qq_bound,usr_info_wechat_bound,usr_info_account_grade,
                   usr_info_dummy],axis=1)

usr_all.index=usr_info_target['id']

#合并bank_card usr_info
bank_usr_0331=pd.merge(usr_all,bank_final,left_index=True,right_index=True,how='inner')
bank_usr_0331.to_csv('bank_usr_0331.csv',index=True,header=True)