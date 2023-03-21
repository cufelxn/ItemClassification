
all_data = []
f1 = open("data.txt")
for line in f1.readlines():
    # print(line.strip())
    tmp_data = []
    tmp_data.append(line.strip())
    tmp_data.append("I")
    all_data.append(tmp_data)
f1.close()

f2 = open("data_c.txt")
for line in f2.readlines():
    tmp_data = []
    tmp_data.append(line.strip())
    tmp_data.append("C")
    all_data.append(tmp_data)
f2.close()

print(len(all_data))
import random
random.shuffle(all_data)

import json
sum_num = len(all_data)
train_data = all_data[: int(sum_num * 0.8)]
train_dict = {}
for id, data in enumerate(train_data):
    tmp_dict = {}
    content = data[0]
    intent = data[1]
    tmp_dict["content"] = content
    tmp_dict["intent"] = intent
    train_dict[str(id)] = tmp_dict
print(len(train_dict))
f_train = open("train.json", "w")
json.dump(train_dict, f_train, ensure_ascii=False)
f_train.close()
print("train.json over")

dev_data = all_data[int(sum_num * 0.8):int(sum_num * 0.9)]
dev_dict = {}
for id, data in enumerate(dev_data):
    tmp_dict = {}
    content = data[0]
    intent = data[1]
    tmp_dict["content"] = content
    tmp_dict["intent"] = intent
    dev_dict[str(id)] = tmp_dict
print(len(dev_dict))
f_dev = open("dev.json", "w")
json.dump(dev_dict, f_dev, ensure_ascii=False)
f_dev.close()
print("dev.json over")


test_data = all_data[int(sum_num * 0.9):]
test_dict = {}
for id, data in enumerate(test_data):
    tmp_dict = {}
    content = data[0]
    intent = data[1]
    tmp_dict["content"] = content
    tmp_dict["intent"] = intent
    test_dict[str(id)] = tmp_dict
print(len(test_dict))
f_test = open("test.json", "w")
json.dump(test_dict, f_test, ensure_ascii=False)
f_test.close()
print("test.json over")

print("all over")