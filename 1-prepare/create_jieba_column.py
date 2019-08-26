# -*- coding: utf-8 -*-
import pandas as pd
import jieba.analyse
import time
import jieba
import jieba.posseg

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',500)

def jieba_cut(text):
    s = []
    str = ""
    #count=0
    words = jieba.posseg.cut(text)  # 带有词性的精确分词模式
    allowPOS = ['n','v','j'] #只取名词，动词，形容词
    for word, flag in words:
        #count +=1
        #print(count)
        #print(word+' '+flag)
        #POS[flag]=POS.get(flag,0)+1
        # 取三个词性的词，并且词的个数>2
        if (flag[0] in allowPOS) and len(word)>=2:
            str += word + " "
    s.append(str)
    final_str = " ".join(s)
    if final_str.__eq__(""):
        return 'N'
    return final_str

def read_text(filename,writefile):
    # 读入训练数据
    train_data2 = pd.read_csv(filename, sep=',', low_memory=False)
    #train_data2 = train_data2.iloc[0:20, :]

    #TODO 进行填充
    train_data2['A_Content'] = train_data2.A_Content.fillna('U')
    train_data2['R_Content'] = train_data2.R_Content.fillna('U')
    train_data2['A_Title'] = train_data2.A_Title.fillna('U')
    train_data2['R_Title'] = train_data2.R_Title.fillna('U')

    train_data2['text_jieba'] = train_data2['A_Title'] + '，' + train_data2['R_Title']+\
                                    train_data2['A_Content'] + '，' + train_data2['R_Content']

    train_data2['jieba'] \
        = train_data2.apply(lambda x: jieba_cut(x['text_jieba']), axis=1)

    train_data2 = train_data2[['jieba']]
    # 生成新的table1
    train_data2.to_csv(writefile, sep=',', header=True, index=False, line_terminator="\n")
def read_one_colunm(filename,writefile,colunm):
    # 读入训练数据
    train_data2 = pd.read_csv(filename, sep=',', low_memory=False)
    #train_data2 = train_data2.iloc[0:20, :]

    #TODO 进行填充
    train_data2['A_Content'] = train_data2.A_Content.fillna('U')
    train_data2['R_Content'] = train_data2.R_Content.fillna('U')
    train_data2['A_Title'] = train_data2.A_Title.fillna('U')
    train_data2['R_Title'] = train_data2.R_Title.fillna('U')

    train_data2[colunm] \
        = train_data2.apply(lambda x: jieba_cut(x[colunm]), axis=1)

    train_data2 = train_data2[[colunm]]
    # 生成新的table1
    train_data2.to_csv(writefile, sep=',', header=True, index=False, line_terminator="\n")

if __name__ == '__main__':
    table_filename = 'test_merge.csv'
    write_to_file_list = ['test_jieba_A_Title.csv','test_jieba_R_Title.csv','test_jieba_A_Content.csv','test_jieba_R_Content.csv']
    colunm_list = ['A_Title','R_Title','A_Content','R_Content']
    for i,j in zip(write_to_file_list,colunm_list):
        read_one_colunm(table_filename,i,j)

