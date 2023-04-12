import numpy as np
from collections import Counter, OrderedDict
#import pandas as pd
#import matplotlib.pyplot as plt

#Be careful with paths!
ORDER = "value" #can change to "by value"

# Related to train
# -------------------------------------------------------
train_dataset = []
with open("trabalho1/archive/Train.csv", "r") as train_stats:
    for line in train_stats:
        if not line.startswith("Width"):
            columns = line.split(",")
            train_dataset.append(int(columns[-2]))

print("Number of images to train:", len(train_dataset))
counter = dict(Counter(train_dataset))
#Eventualy draw this on plot
if ORDER == "key":
    for key in sorted(counter.keys()):
        print(key, counter[key])
if ORDER == "value":
    keys = list(counter.keys())
    values = list(counter.values())
    sorted_value_index = np.argsort(values)[::-1]
    print("\nSign id  number of images")
    for i in sorted_value_index:
        print("{:2}\t{}".format(keys[i], values[i]))

# Related to test
# -------------------------------------------------------
test_classes = []
with open("trabalho1/archive/Test.csv", "r") as test_stats:
    for line in test_stats:
        if not line.startswith("Width"):
            columns = line.split(",")
            test_classes.append(int(columns[-2]))

print("Number of images to test:", len(test_classes))
counter = dict(Counter(test_classes))
#Eventualy draw this on plot
if ORDER == "key":
    for key in sorted(counter.keys()):
        print(key, counter[key])
if ORDER == "value":
    keys = list(counter.keys())
    values = list(counter.values())
    sorted_value_index = np.argsort(values)[::-1]
    print("\nSign id  number of images")
    for i in sorted_value_index:
        print("{:2}\t{}".format(keys[i], values[i]))