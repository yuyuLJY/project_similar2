import pandas as pd

f1 = pd.read_csv('../submit_example.csv', sep=',', low_memory=False)
f1.drop(columns=['Level'],inplace=True)
print(f1.info())

f2 = pd.read_csv('StackingSubmission.csv', sep=',', low_memory=False)
print(f2.info())
merge = pd.merge(f1,f2,how='left')
print(merge.info())

merge.to_csv('sort.csv', sep=',', header=False, index=False, line_terminator="\n")