import csv,sys

# if it contains in a[] then remove 12345689, extra
def check_int_contain (input_str):
    a=['<','>','(',")",'\\',':','.',"'",'*','-','&','1','2','3','4','5','6','7','8','9','0','!','?','it']
    for i in range(0, len(a)-1):
        if a[i] in input_str:
            return ""
    return input_str

# check whethere it == the this that
def check_normal_same (input_str):
    a=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","%","=",",","@","!",'as',"'",'in','a','is','are','this','that','it','if','were','was','them','it','to']
    for i in range(0, len(a)-1):
        if a[i]== input_str:
            return ""
    return input_str



def ham_spam_list(rdr):
    words_ham=[]
    words_spam=[]

    for line in rdr:
        if (line[0]=='ham'):# if it is ham, then add to (all)list
            temp=line[1].split(" ")
            for i in range(0,len(temp)):
                # check whethere it contains 123456789, extra char like @#$ the this that
                t=temp[i]
                t=t.lower()
                t = check_int_contain(t)
                t = check_normal_same(t)
                temp[i] = t
            words_ham.append(temp)
        elif (line[0]=='spam'):# if it is ham, then add to (all)list
            temp=line[1].split(" ")
            for i in range(0,len(temp)):
                # check whethere it contains 123456789, extra char like @#$ the this that
                t=temp[i]
                t=t.lower()
                t = check_int_contain(t)
                t = check_normal_same(t)
                temp[i] = t
            words_spam.append(temp)

    result_ham = []
    for i in range(0, len(words_ham)):
        result_ham.extend(words_ham[i])# all list to one list
    result_ham = list(set(result_ham))# remove same things
    result_ham.remove("")

    result_spam = []
    for i in range(0, len(words_spam)):
        result_spam.extend(words_spam[i])# all list to one list
    result_spam = list(set(result_spam))# remove same things
    result_spam.remove("")

    return (result_ham,result_spam)

if __name__=="__main__":
    f = open('spam_test.csv', 'r')
    rdr = csv.reader(f)
    ham,spam = ham_spam_list(rdr)
    print("ham ",(len(ham)+1) )
    print(ham)
    print("spam ", (len(spam)+1))
    print(spam)
    f.close()






