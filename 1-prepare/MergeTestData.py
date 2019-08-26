import pandas as pd

data1 = pd.read_csv('new_Test_Achievements.csv',sep=",",encoding="utf-8")
print(data1.info())
print('*'*60)

data2 = pd.read_csv('new_Requirements.csv',sep=",",encoding="utf-8")
print(data2.info())
print('*'*60)

data3 = pd.read_csv('../TestPrediction.csv',sep=",",encoding="utf-8")
print(data3.info())
print('*'*60)

merge = pd.merge(data3,data1,how='left')
merge = pd.merge(merge,data2,how='left')

print(merge.info())
print('*'*60)
merge.to_csv('test_merge.csv', sep=',', header=True, index=False, line_terminator="\n")