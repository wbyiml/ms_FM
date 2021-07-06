import os
import random


# os.makedirs('trainset', exist_ok=True)

# train_percent = 0.9

# datafile = open('train.txt','r', errors='ignore')
# datas = datafile.readlines()

# numdatas = len(datas)
# listdatas = range(numdatas)

# numtrain = int(numdatas * train_percent)
# listtrain = random.sample(listdatas, numtrain)

# trainfile = open(os.path.join('trainset/train.txt'),'w')
# testfile = open(os.path.join('trainset/test.txt'),'w')

# for i in listdatas:
#     if i in listtrain: # train
#         trainfile.writelines(datas[i])
#     else: # test
#         testfile.writelines(datas[i])



os.makedirs('midset', exist_ok=True)

train_percent = 0.9

datafile = open('set/train.txt','r', errors='ignore')
datas = datafile.readlines()[:1000000]

numdatas = int(len(datas)/10)
listdatas = range(numdatas)

numtrain = int(numdatas * train_percent)
listtrain = random.sample(listdatas, numtrain)

trainfile = open(os.path.join('midset/train.txt'),'w')
testfile = open(os.path.join('midset/test.txt'),'w')

for i in listdatas:
    if i in listtrain: # train
        trainfile.writelines(datas[i*10:i*10+10])
    else: # test
        testfile.writelines(datas[i*10:i*10+10])
