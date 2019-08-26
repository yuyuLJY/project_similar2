# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import csv

#TODO 处理表1

# 表1：Achievements
#add = 'new_Train_Achievements'
#fileName = '../Train_Achievements.csv'
#column_list = ['A_ID', 'A_Title', 'A_Content']

# 表2：Requirements
#add = 'new_Requirements'
#fileName = '../Requirements.csv'
#column_list = ['R_ID', 'R_Title', 'R_Content']

# 表1：test Achievements
add = 'new_Test_Achievements'
fileName = '../Test_Achievements.csv'
column_list = ['A_ID', 'A_Title', 'A_Content']

csvfile = open(add + '.csv', 'w',encoding='utf8')# the path of the generated train file
writer = csv.writer(csvfile)
writer.writerow(column_list)
with open(fileName, 'r',encoding='utf8',newline='') as f:
    count =1
    for line in f:
        line = line.strip().replace("'", '').replace("\\n", '')
        data = line.split(" , ")
        #将data的第三列组合起来
        for j in range(2,len(data)):
            data[2] = data[2]+data[j]
        data = data[0:3]
        if len(data) != 3:
            print(count)
            print(data)
        writer.writerow(data)
        count += 1


#TODO 处理表3
# 表3：Requirements
'''
add = 'new_Train_Interrelation'
fileName = '../Train_Interrelation.csv'
column_list = ['L_ID','A_ID','R_ID','Level']
csvfile = open(add + '.csv', 'w',encoding='utf8')# the path of the generated train file
writer = csv.writer(csvfile)
writer.writerow(column_list)
with open(fileName, 'r',encoding='utf8',newline='') as f:
    count =1
    for line in f:
        line = line.strip().replace("'", '').replace("\\n", '')
        data = line.split(",")
        if len(data) != 4:
            print(count)
            print(data)
        writer.writerow(data)
        count += 1

'''