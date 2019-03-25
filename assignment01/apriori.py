import sys
from itertools import combinations, chain

# read file and returns list containing transactions
def loadDatabase(inputfile):
    f = open(inputfile, mode='rt')
    data = []
    while True:
        line = f.readline()
        if not line:
            break
        line = line[:-1]
        items = []
        for i in line.split('\t'):
            items.append(int(i))
        data.append(set(items))
    f.close()
    return data

# calculate support value for each candidate in Ck 
# and returns frequent itemsets
def scanDatabase(D, Ck, minSupport):
    supportCnt = {}
    
    for cand in Ck:
        for trans in D:
            if cand.issubset(trans):
                if not cand in supportCnt:
                    supportCnt[cand] = 1
                else: 
                    supportCnt[cand] += 1
    transNum = float(len(D))
    Lk = []                 # frequent itemsets of size k
    supportData = {}        # support data
    for key, value in supportCnt.items():
        s = value / transNum * 100
        if s >= float(minSupport):
            Lk.append(key)
        supportData[key] = s
    return Lk, supportData                    

# generate candidate itemset of size 1 by 
# scanning through database
def createC1(D):
    C1 = []             # candidate itemsets of size 1
    for trans in D:
        for item in trans:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset, C1)

# generate candidate itemsets of length k 
# from frequent itemsets
def generateCandidate(Lk, k):
    retList =  []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = Lk[i]; L2 = Lk[j]
            if len(L1 & L2) == k-2:
                if not (L1 | L2) in retList:
                    retList.append(L1 | L2)
    return retList

# apriori algorithm
def apriori(D, minSupport):
    C1 = createC1(D)
    L1, supportData = scanDatabase(D,C1,minSupport)
    L = [L1]
    k = 2
    while(len(L[k-2]) > 0):
        Ck = generateCandidate(L[k-2], k)
        Lk, sup = scanDatabase(D, Ck, minSupport)
        supportData.update(sup)
        L.append(Lk)
        k += 1
    return L[:-1],supportData

# function that returns the subsets of a set
# parameters      
def getSubsets(arr):
    return list(chain(*[combinations(arr, i+1) for i in range(len(arr))]))

# generate association rules
def generateRules(L,supportData):
    retRules = []
    for Lk in L[1:]:
        for item in Lk:
            subsets = map(frozenset, [x for x in getSubsets(item)])
            for element in subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = supportData[item] / supportData[element] * 100
                    retRules.append(((element, remain), confidence))
    return retRules


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("need 3 arguments")
    else:
        # Execute the program with three arguments.
        #    minimum support 
        #    input file name
        #    output file name
        minSupport = float(sys.argv[1])
        inputfile = sys.argv[2]
        outputfile = sys.argv[3]

        fout = open(outputfile, 'w')

        database = loadDatabase(inputfile)
        L, supportData = apriori(database, minSupport)
        assocRules = generateRules(L, supportData)
        
        for (a, b), conf in assocRules:
            sup = supportData[a | b]
            fout.write("{}\t{}\t".format(set(a), set(b)))
            fout.write("{0:.2f}\t".format(sup))
            fout.write("{0:.2f}\n".format(conf))

        fout.close()
