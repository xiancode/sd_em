#!/usr/bin/env  Python
#-*- coding=utf-8 -*-

'''
Description : statistic indicator area ranks by entropy method
require     : windows Anaconda-2.3.0
author      : shizhongxian@126.com
usage  $python sd_em.py  -f table.txt 
'''

import pandas as pd
import numpy as np
import math
import logging
import sys
import os
import string
from optparse import OptionParser

#global area_list

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename="C:\\LOG\\entropy_method.log",
                    filemode='a')


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
    indicators = pivoted.columns
    #删除空值行
    cleaned_data = pivoted.dropna(axis=0)
    area_list = cleaned_data.index
    return cleaned_data,area_list,indicators

def sd_em(fname,result_name):
    '''
    entropy method计算   http://blog.sina.com.cn/s/blog_6163bdeb0102dvow.html
    '''
    cl_data,area_list,indicators = data_set(fname)
    origin_values = cl_data.values
    indi_list = indicators.tolist()
    logging.info("cleaned  area:"+" ".join(area_list))
    logging.info("selected indi:"+" ".join(indi_list))

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
    print g
    w = g/sum(g)
    scores = np.dot(origin_values,w.T)
    scores_list = scores.tolist()
    
    
    #输出
    display_e_ = map(lambda x:str(round(x,4)),e_)
    dispaly_w  = map(lambda x:str(round(x,4)),w)
    
    #
    fout = open(result_name,"w")
    
    fout.write("指标\t")
    fout.write("\t".join(indi_list))
    fout.write("\n")
    
    fout.write("指标熵值\t")
    fout.write("\t".join(display_e_))
    fout.write("\n")
    
    fout.write("指标权重\t")
    fout.write("\t".join(dispaly_w))
    fout.write("\n")
    
    
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
    print "save to ",result_name
    
if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv convert from rec',
                         default=None)
    (options, args) = optparser.parse_args()
    #inFile = None
    
    if options.input is None:
            inFile = sys.stdin
            #inFile = "INTEGRATED-DATASET.csv"
    elif options.input is not None:
            inFile = options.input
    else:
            print 'No dataset filename specified, system with exit\n'
            sys.exit('System will exit')

    full_name = os.path.realpath(inFile)
    pos = full_name.find(".txt")
    result_name = full_name[:pos] + "_result.txt"
    
    #table_file_name = "table.txt"
    #sdtool.rec2table("2013_10.txt", table_file_name)
    sd_em(full_name,result_name)
    
    
    
    
    
    


