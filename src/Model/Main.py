import tensorflow as tf

import csv,sys


def check_int_contain (input_str):
    a=['+','=','<','>','(',")",'\\',':','.',"'",'*','-','&','1','2','3','4','5','6','7','8','9','0','!','?','it']
    for i in range(0, len(a)-1):
        if a[i] in input_str:
            return ""
    return input_str


def check_normal_same (input_str):
    a=["=",",","@","!",'as',"'",'in','a','is','are','this','that','it','if','were','was','them','it','to']
    for i in range(0, len(a)-1):
        if a[i]== input_str:
            return ""
    return input_str




if __name__=="__main__":
    f = open('spam_test.csv', 'r')
    rdr = csv.reader(f)
    all=[]
    all_spam=[]
    spam_str=[]
    words=[]
    count=0

    for line in rdr:
      part=[]
      if (line[0]=='spam'):
        all.append(line)
        all_spam.append(line[1])
        temp=line[1].split(" ")
        for i in range(0,len(temp)):
            t=temp[i]
            t=t.lower()
            t=check_int_contain(t)
            t = check_normal_same(t)
            temp[i]=t

        words.append(temp)
      count=count+1
      #print(line)
#      if count==5000:
#          break


    #print(all)
    print(all_spam)
    print("words")
    print(words)
    words_str=[]
    for i in range(0,len(words)):
        words_str.extend(words[i])
    words_str=list(set(words_str))
    print("words_str")

    print(words_str)
    print(len(words_str))


    f.close()






