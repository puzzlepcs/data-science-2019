# author: 2016024793 김유진
# Data Science assignment #2: Decision Tree

import math
import sys
from collections import Counter, defaultdict
from functools import partial

# read train data file
# return list of attributes and following categories. 
def load_train_data(filename):
    f = open(filename, "r")
    attr = []
    cat = []
    data = []

    attr = f.readline().strip("\n").split("\t")
    for _ in range(len(attr)):
        cat.append([])
    while True:
        line = f.readline()
        if not line:
            break
        t = line.strip("\n").split("\t")
        tmp = dict()

        for i in range(len(attr)):
            if not t[i] in cat[i]:
                cat[i].append(t[i])
            if i != len(attr)-1:
                tmp[attr[i]] = t[i]

        data.append((tmp, t[-1]))

    f.close()
    return data, attr, cat

def load_test_data(filename):
    f = open(filename, "r")
    attr = []
    data = []

    attr = f.readline().strip("\n").split("\t")
    while True:
        line = f.readline()
        if not line:
            break
        
        t = line.strip("\n").split("\t")
        tmp = dict()
        for i in range(len(t)):
            tmp[attr[i]] = t[i]
        data.append(tmp)

    f.close()
    return data

# return entropy value using list of probabilities
def entropy(probabilities):
    return sum(-p * math.log(p, 2) for p in probabilities.values() if p is not 0)

# return a dictionary of probabilities by counting labels 
#  for example, {'no':0.5, 'yes':0.5}
def class_probabilities(labels):
    total_cnt = len(labels)
    return dict([(key, value / total_cnt) for key, value in Counter(labels).items()])

# return information gain of labeled dataset
def info_gain(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

# returns information gain of subsets
def partition_info_gain(subsets):
    total_cnt = sum(len(subset) for subset in subsets)
    return sum(info_gain(subset) * len(subset) / total_cnt for subset in subsets)

# returns split information of subsets.
def partition_split_info(subsets):
    total_cnt = sum(len(subset) for subset in subsets)
    return sum(-1 * len(s) / total_cnt * math.log(len(s)/total_cnt, 2) for s in subsets)

def partition_by(inputs, attribute):
    # divide inputs into subsets by the attribute
    groups = defaultdict(list)
    for i in inputs:
        key = i[0][attribute]
        groups[key].append(i)
    return groups

# return information gain when splited by given attribute
def partition_info_gain_by(inputs, attribute):    
    partitions = partition_by(inputs, attribute)
    infoD = info_gain(inputs)
    return infoD - partition_info_gain(partitions.values())

# return gain ratio when splited by given attribute
def partition_gain_ratio_by(inputs, attribute):
    partitions = partition_by(inputs, attribute)
    gain = partition_info_gain_by(inputs, attribute)
    return gain / partition_split_info(partitions.values())

# build tree with training dataset
def build_tree(inputs, split_candidates=None):
    def majority(num_class):
        maxKey = None
        maxVal = -1
        for key, value in num_class.items():
            if maxVal < value:
                maxKey, maxVal = key, value
        return maxKey

    if split_candidates is None:
        split_candidates = inputs[0][0].keys()
    
    num_inputs = len(inputs)
    num_class = Counter([label for _,label in inputs])

    # stop conditions
    # 1) if all tuples have same class labels, return that class label
    for key, value in num_class.items():
        if value == num_inputs:
            return key
    
    # 2) if there is no more attributes to select,
    #    return a class label by majority voting
    if not split_candidates:
        return majority(num_class)

    # choose best attribute that has largest information gain
    # best_attr = max(split_candidates, key=partial(partition_info_gain_by, inputs))
    
    # choose best_attribute that has largest gain ratio 
    best_attr = max(split_candidates, key=partial(partition_gain_ratio_by, inputs))
    partitions = partition_by(inputs, best_attr)
    new_candidates = [a for a in split_candidates if a != best_attr]

    # build tree recursively
    subtrees = {attr_value : build_tree(subset, new_candidates) 
                    for attr_value, subset in partitions.items()}
    # for exception handling
    subtrees[None] = majority(num_class)
    return(best_attr, subtrees)

# classify test dataset recursively 
def classify(tree, input, labels):
    # if it is a leaf node, return a value
    if tree in labels:
        return tree
    
    # otherwise, reach to next node 
    attr, subtree_dict = tree
    subtree_key = input.get(attr)

    if subtree_key not in subtree_dict:
        subtree_key = None

    subtree = subtree_dict[subtree_key]

    return classify(subtree, input, labels)        

def print_result(tree, testData, labels, filename):
    f = open(filename, 'w')
    
    for a in attr:
        f.write("{}\t".format(a))
    f.write('\n')

    for t in testData:
        for value in t.values():
            f.write("{}\t".format(value))
        result = classify(tree, t, labels)
        f.write("{}\n".format(result))
    f.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("need 3 arguments")
    else:
        trainfile = sys.argv[1]
        testfile = sys.argv[2]
        outputfile = sys.argv[3]

        data, attr, cat = load_train_data(trainfile)
        tree = build_tree(data)
        testData = load_test_data(testfile)
        print_result(tree, testData, cat[-1], outputfile)
    