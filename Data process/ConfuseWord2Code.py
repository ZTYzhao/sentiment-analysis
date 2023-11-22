import string
import re
import numpy as np
import xlrd
import os
import shutil



def FenJuZi(colnames,content_lists):



    numJuZi = len(colnames)
    for i in range(numJuZi):
        Codelist_list=[]
        print('正在处理第'+str(i)+'/'+str(numJuZi))
        #sentence = colnames[i].rsplit(".")
        sentence = re.split('[.?]',colnames[i])
        sentence_orginal = sentence[:]
        sentence_orginal2 = sentence[:]



        for sentenc in sentence:
            if sentenc=='\xa0' or sentenc=='' or sentenc==' ':
                sentence_orginal2.remove(sentenc)
        sentence_final = sentence_orginal2
        JuZiNumber = len(sentence_final)
       # with open('Word2Code_test.txt', 'a+', encoding='utf-8') as file0:
            #print(str(JuZiNumber), file=file0)
        for j in range(len(sentence_final)):
            #with open('Word2Code_test.txt', 'a+', encoding='utf-8') as file0:
                #print('\n',file=file0)
            #print("第"+str(i)+"句子数量")
            sentence_final[j] = sentence_final[j].translate(sentence_final[j].maketrans('', '', string.punctuation))#剔除标点符
            sentence_final[j] = sentence_final[j].strip()#剔除空白符
            if '?' in sentence_final[j]:
                sentence_final[j] = sentence_final[j]+' '+'?'

            str_words = sentence_final[j].rsplit(' ')
            Codelist = []
            for str_word in str_words:
                if str_word.isalpha():
                    if str_word.lower() in [content_list.lower() for content_list in content_lists]:
                        try:
                            Code = content_lists.index(str_word)
                            Code = Code+1
                            Codelist.append(str(Code))
                        except:
                            continue

            #str_word = [s.lower() for s in str_word if isinstance(s, str) == True]
            print(sentence[j])
            while " " in Codelist:
                Codelist.remove(" ")
            print(Codelist)

            if Codelist != []:
                Codelist_list.append(Codelist)
        if len(Codelist_list)!=0:
            with open('Word2Code_medicine.txt','a+',encoding='utf-8') as file0: #Word2Code_medicine.txt Word2Code_edu.txt Word2Code_human.txt
                print(str(len(Codelist_list)),file=file0)
                for col in Codelist_list:
                    for single in col:
                        #if single!=col[-1]:
                        file0.write(single + ' ')
                        #else:
                            #file0.write(single)
                    file0.write('\n')
            # with open('Word2Code_test.txt','a+',encoding='utf-8') as file0:
            #     print(str(len(Codelist_list)), file=file0)
            #     #print('\n',file=file0)
            #     for single in Codelist:
            #
            #         if single!=Codelist[-1]:
            #               file0.write(single + ' ')
            #         else:
            #             file0.write(single)
            #     file0.write('\n')

def get_by_index(filename,colindex,by_index):
    data = xlrd.open_workbook(filename)
    table = data.sheets()[by_index]
    colnames = table.col_values(colindex)
    return colnames

def CreatedWordList(excel_path): #根据excel中评论创建对应wordlist

    WordList = []

    colnames = get_by_index(excel_path, 1, 0)

    numJuZi = len(colnames)
    i = 0

    for i in range(numJuZi):
        try:
            sentence = colnames[i].rsplit(".\xa0")
        except:
            print('error1:',i)
        JuZiNumber = len(sentence)
        j = 1
        for j in range(len(sentence)):
            # print("第"+str(i)+"句子数量")
            sentence[j] = sentence[j].translate(sentence[j].maketrans('', '', string.punctuation))  # 剔除标点符
            sentence[j] = sentence[j].strip()  # 剔除空白符
            #sentence[j] = [s.lower() for s in sentence[j] if isinstance(s, str) == True]#将大写字母变小写，整型数剔除
            str_word = sentence[j].rsplit(' ')
            str_word = [s.lower() for s in str_word if isinstance(s, str) == True]
            #print(sentence[j])
            k = 0
            for k in range(len(str_word)):
                if str_word[k] not in WordList:
                  WordList.append(str_word[k])
    WordList.sort()




def Word2Code():
    with open('E:\pythonProject\Data process/words.txt', encoding='utf-8') as file:  #ASUM-WordList Stanf-wordlist words
        content = file.read()
        content_lists = content.rsplit("\n")

    colnames = get_by_index(excel_path,1,0)
    FenJuZi(colnames, content_lists)
    return


if __name__ == '__main__':

    excel_path = 'E:\pythonProject\Data process\confusion-medicine.xlsx' #confusion-medicine.xlsx confusion-edu.xlsx confusion-human.xlsx

    Word2Code()

    #CreatedWordList(excel_path)
