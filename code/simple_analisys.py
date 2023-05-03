import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict


# Related to train
# -------------------------------------------------------
train_dataset = []
train_dimensions = []
with open("trabalho1/archive/Train.csv", "r") as train_stats:
    for line in train_stats:
        if not line.startswith("Width"):
            columns = line.split(",")
            train_dataset.append(int(columns[-2]))
            train_dimensions.append(int(columns[0]) * int(columns[0]))

train_dimensions = np.array(train_dimensions)
by_pixels = []
by_pixels.append((train_dimensions <= 40*40).sum())
by_pixels.append((train_dimensions <= 80*80).sum())
by_pixels.append((train_dimensions <= 100*100).sum())
by_pixels.append((train_dimensions > 100*100).sum())
print((train_dimensions <= 40*40).sum())
print((train_dimensions <= 80*80).sum())
print((train_dimensions <= 100*100).sum())
print((100 * 100 < train_dimensions).sum())
#fig, ax = plt.subplots()
#ax.pie(by_pixels, labels=["less than 40x40", "less than 80x80", "less than 100x100", "more than 100x100"])
#fig.show()

print("Number of images to train:", len(train_dataset))
counter = dict(Counter(train_dataset))
keys = list(counter.keys())
values = list(counter.values())
sorted_value_index = np.argsort(values)[::-1]
for i in sorted_value_index:
    #print("{:2}\t{}".format(keys[i], values[i]))
    continue
values.sort(reverse=True)
plt.bar(range(0,43), values)
plt.xticks(range(0,43), sorted_value_index)
plt.title("Number of samples per class on train dataset")
plt.xlabel("class ID")
plt.ylabel("samples")
#plt.show()


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
keys = list(counter.keys())
values = list(counter.values())
#sorted_value_index = np.argsort(values)[::-1]
for i in sorted_value_index:
    #print("{:2}\t{}".format(keys[i], values[i]))
    continue
values.sort(reverse=True)
print(values)
plt.bar(range(0,43), values)
plt.xticks(range(0,43), sorted_value_index)
plt.title("Number of samples per class on test dataset")
plt.xlabel("class ID")
plt.ylabel("samples")
plt.show()

print(len(test_classes) / (len(train_dataset) + len(test_classes)) * 100)
