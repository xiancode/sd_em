#!/usr/bin/env  Python
#-*- coding=utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn import preprocessing
from scipy import linalg
from sdtool import  sdtool
import math

#global area_list

def zeor_one_norm(values):
    '''
    0-1 normalization translate
    '''
    v_max = values.max(axis=0)
    v_min = values.min(axis=0)
    diiff = v_max - v_min
    result = (values  - v_min)/diiff + 1
    return result
    

def data_set(fname):
    '''
        把数据转化为二维表格,每行表示一个时间段,每列表示一个指标
        删除包含空值的行
    '''
    df = pd.read_csv(fname,"\t")
    #data = df.rename(columns={'月份顺序排序':'m_order','正式指标':'indicator','正式数值':'value'})
    data = df.rename(columns={'地区':'area','正式指标':'indicator','正式数值':'value'})
    pivoted = data.pivot('area','indicator','value')
    #删除空值行
    cleaned_data = pivoted.dropna(axis=0)
    area_list = pivoted.index
    return cleaned_data,area_list

def sd_em(fname,components=2):
    '''
    entropy method计算   http://blog.sina.com.cn/s/blog_6163bdeb0102dvow.html
    '''
    cl_data,area_list = data_set(fname)
    origin_values = cl_data.values

    #数据标准化
    values = zeor_one_norm(origin_values)
    #calculate p
    s_0 = values.sum(axis=0) 
    p = values/s_0
    #calculate Ee
    n,m = values.shape
    k = 1/math.log(n,math.e)
    e = (-k)*p*np.log(p)
    e_ = e.sum(axis=0)
    Ee = np.sum(e_)
    g = (1-e_)/(m-Ee)
    #calculate w
    w = g/sum(g)
    scores = np.dot(origin_values,w.T)
    scores_list = scores.tolist()
    
    #
    fout = open("em_result.txt","w")
    fout.write("\n===============================\n")
    if len(area_list) == len(scores):
        area_scores = zip(scores_list,area_list)
        as_dict = dict((key,value) for key,value in area_scores)
        #order by scores
        #scores.sort
        scores_list.sort(reverse=True)
        for score in scores_list:
            #print area_list[i],scores[i]
            fout.write("%s,%.5f \n" % (as_dict[score],score))
    else:
        print "caculated result not equal to area_list"
    fout.close()
    print "save to em_result.txt"
    
if __name__ == "__main__":
    table_file_name = "table.txt"
    sdtool.rec2table("2013_10.txt", table_file_name)
    sd_em(table_file_name)
    
    
    
    
    
    


