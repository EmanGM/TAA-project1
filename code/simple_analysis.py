import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict


# Related to train
# -------------------------------------------------------
train_dataset = []
train_dimensions_x = []
train_dimensions_y = []
with open("trabalho1/archive/Train.csv", "r") as train_stats:
    for line in train_stats:
        if not line.startswith("Width"):
            columns = line.split(",")
            train_dataset.append(int(columns[-2]))
            train_dimensions_x.append( int(columns[0]))
            train_dimensions_y.append( int(columns[1]))

train_dimensions = np.array(train_dimensions_y)
by_pixels = []
by_pixels.append((train_dimensions <= 30).sum())
by_pixels.append(len(([1 for i in train_dimensions if 30 < i <= 40])))
by_pixels.append(len(([1 for i in train_dimensions if 40 < i <= 60])))
by_pixels.append(len(([1 for i in train_dimensions if 60 < i <= 80])))
by_pixels.append(len(([1 for i in train_dimensions if 80 < i <= 100])))
by_pixels.append(len(([1 for i in train_dimensions if i > 100])))


plt.bar(["< 30", "< 40", "< 60", "< 80", "< 100", "> 100"], by_pixels, color='orange')
plt.legend()
plt.title("Width of the images")
plt.xlabel("width in pixels")
plt.ylabel("number of samples")
plt.show()

print("Number of images to train:", len(train_dataset))
counter = dict(Counter(train_dataset))
keys = list(counter.keys())
values = list(counter.values())
sorted_value_index = np.argsort(values)[::-1]
for i in sorted_value_index:
    #print("{:2}\t{}".format(keys[i], values[i]))
    continue
values.sort(reverse=True)
print(values)


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
test_values = list(counter.values())
#sorted_value_index = np.argsort(values)[::-1]
for i in sorted_value_index:
    #print("{:2}\t{}".format(keys[i], values[i]))
    continue
test_values.sort(reverse=True)
print(test_values)
print(len(values), len(test_values))
va = np.array(values)
vb = np.array(test_values)
plt.bar(range(0,43), va + vb, label="train")
plt.bar(range(0,43), vb, label="test")
plt.legend()
plt.xticks(range(0,43), sorted_value_index)
plt.title("Number of samples per class on dataset")
plt.xlabel("class ID")
plt.ylabel("samples")
plt.show()

print(len(test_classes) / (len(train_dataset) + len(test_classes)) * 100)
