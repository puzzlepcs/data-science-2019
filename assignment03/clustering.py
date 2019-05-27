'''
@ author: 2016024793 김유진
DataScience Assignment #3: DBSCAN implementation
'''

import math
import sys

# Function that read inputfile and save the data. 
def readFile(filename):
    f = open(filename, "r")
    D = list()
    while True:
        line = f.readline()
        if not line:
            break
        object_id, x_coord, y_coord = line.strip('\n').split('\t')
        tmp = (int(object_id), float(x_coord), float(y_coord))
        D.append(tmp)
    f.close()
    return D

# Function that returns top n clusters with respect of its size.
def saveOutput(D, labels, n):
    clusters = dict()
    for i in range(len(D)):
        try: 
            obj_id, _, _ = D[i]
            label = labels[i]
            clusters[label].append(obj_id)
        except:
            clusters[label] = [obj_id]
    tmp = list()
    for label, ids in clusters.items():
        # print("cluster {}: {}".format(label, len(ids)))
        if label == -1:
            continue
        tmp.append((label, len(ids)))
    tmp.sort(key=lambda tuple: tuple[1])
    
    output = list()
    for label, _ in tmp[-n:]:
        output.append(clusters[label])
    return output

# Function that prints the result
def printOutput(filename, output):
    filename = filename.split('/')[-1]
    outputfile = filename.split('.')[0]
    for i in range(len(output)):
        tmpfile = "{}_cluster_{}.txt".format(outputfile, i)
        f = open(tmpfile,"w")
        for o in output[i]:
            f.write("{}\n".format(o))
        f.close()

# Function for calculating distance between two objects.
def distance(p1, p2):
    _, x1, y1 = p1
    _, x2, y2 = p2
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

# Function for finding neighbors.
# Scans whole dataset and returns list of objects of which 
# distance to object P is smaller than eps.
def findNeighbors(D, P, eps):
    neighbors = []
    for Pn in range(len(D)):
        if distance(D[P], D[Pn]) < eps:
            neighbors.append(Pn)
    return neighbors 

# Function that finds members of cluster that object P is 
# the core point.
def growCluster(D, labels, P, neighbors, C, eps, MinPts):
    # Mark object P as a member of cluster C 
    labels[P] = C
    
    i = 0
    while i < len(neighbors):  
        Pn = neighbors[i]           # check object Pn   
        # If object Pn is marked as an outlier, mark it as cluster C
        if labels[Pn] == -1:        
            labels[Pn] = C
        # Else if object Pn is not marked, mark it as cluster C
        elif labels[Pn] == 0:
            labels[Pn] = C
            PnNeighbors = findNeighbors(D, Pn, eps)
            # If object Pn is not a border point, 
            # add Pn's neighbors to neighbors of object P(Grow cluster C).
            if len(PnNeighbors) >= MinPts:
                for tmp in PnNeighbors:
                    if tmp not in neighbors:
                        neighbors.append(tmp)
        i += 1

# Density-based spatial clustering of applications with noise
# - Labels
#   0 : not labeled yet
#  -1 : outlier
#   1~: cluster numbering
def DBSCAN(D, eps, MinPts):
    labels = [0]*len(D)
    C = 0
    print('Start DBSCAN')
    for P in range(0, len(D)):
        # If P is already labeled, continue.
        if not (labels[P] == 0):
            continue
        
        neighbors = findNeighbors(D, P, eps)
        # If number of P's neighbors are smaller then MinPts, 
        # lable it as an outlier. 
        if len(neighbors) < MinPts:
            labels[P] = -1
        # Else, grow cluster C. After this process, cluster C will 
        # be fully grown!
        else:
            C += 1
            growCluster(D, labels, P, neighbors, C, eps, MinPts)
    print('Clustering Done! {} clusters in total.'.format(C))
    return labels

if __name__=="__main__":
    '''
    input1.txt : n=8, eps=15, minPts=22
    input2.txt : n=5, eps= 2, minPts= 7
    input3.txt : n=4, eps= 5, minPts= 5
    '''
    if len(sys.argv) != 5:
        print("need 4 arguments")
    else:
        inputfile = sys.argv[1]
        n = int(sys.argv[2])
        eps = int(sys.argv[3])
        minPts = int(sys.argv[4])

        D = readFile(inputfile)
        labels = DBSCAN(D, eps, minPts)
        output = saveOutput(D, labels, n)
        printOutput(inputfile, output)

