import pandas as pd

data1 = pd.read_csv('new_Train_Achievements.csv',sep=",",encoding="utf-8")
print(data1.info())
print('*'*60)

data2 = pd.read_csv('new_Requirements.csv',sep=",",encoding="utf-8")
print(data2.info())
print('*'*60)

data3 = pd.read_csv('new_Train_Interrelation.csv',sep=",",encoding="utf-8")
print(data3.info())
print('*'*60)

merge = pd.merge(data3,data1)
merge = pd.merge(merge,data2)

print(merge.info())
print('*'*60)
merge.to_csv('train_merge.csv', sep=',', header=True, index=False, line_terminator="\n")