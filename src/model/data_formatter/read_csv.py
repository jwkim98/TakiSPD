import csv
import os

ignore_list = ['+', '=', '<', '>', '(', ")", '\\', ':', '.', "'", '*', '-', '&', '1', '2', '3', '4', '5', '6', '7',
                   '8', '9', '0', '!', '?', 'it', "=", ",", "@", "!", 'as', "'", 'in', 'a', 'is', 'are', 'this', 'that',
                   'it', 'if', 'were', 'was', 'them', 'it', 'to']


def check_valid(string):
    if any(not c.isalpha() for c in string) or string in ignore_list:
        return False
    return True


def read_csv(filename):

    with open(filename, encoding='latin-1') as csvfile:
        spam_reader = csv.reader(csvfile)
        mail_list = []
        longest = 0
        for row in spam_reader:
            sentence = row[1]
            word_list = sentence.split(" ")
            clean_word_list = [elem for elem in word_list if check_valid(elem)]

            if len(clean_word_list) > longest:
                longest = len(clean_word_list)

            if row[0] == 'ham':
                word_tuple = (clean_word_list, 0.0)
            else:
                word_tuple = (clean_word_list, 1.0)
            mail_list.append(word_tuple)

    return mail_list[1:], longest


