'''
@ author: 2016024793 김유진
DataScience Term Project: Predicting movie ratings
- Utility functions 
'''
import numpy as np

def readFile(filename):
    '''
    Read the file and return a list 
    '''
    f = open(filename, "r")
    D = list()
    while True:
        line = f.readline()
        if not line:
            break
        usr_id, item_id, rating, _ = line.strip('\n').split('\t')
        tmp = [int(usr_id), int(item_id), float(rating)]
        D.append(tmp)
    f.close()
    return D

def toMat(data):
    '''
    Make rating matrix using data
    rows represent users, and columns represent items
    '''
    users, items, ratings = [], [], []
    for u, i, r in data:
        users.append(int(u))
        items.append(int(i))
        ratings.append(r)
    num_users = (max(set(users)))
    num_items = (max(set(items)))
    A = np.zeros((num_users, num_items)).astype(np.float64)

    # print("{} users, {} movies".format(num_users, num_items))
    for u, i, r in data:
        try:
            A[int(u)-1][int(i)-1] = r
        except:
            print("u: {}, i: {}".format(u,i))
    return A

def printResult(filename, result):
    '''
    Print the predictions on a file
    '''
    filename = filename.split('/')[-1]
    outputfilename = filename + '_prediction.txt'
    f = open(outputfilename, "w")
    for user_id, item_id, rating in result:
        f.write('{}\t{}\t{}\n'.format(user_id, item_id, rating))
    f.close()
