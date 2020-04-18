#!/usr/bin/env python
import numpy as np
import re

data_file=open('../data/dataset.txt','r')
data=data_file.read().split('\n')

#Creating training and development data from the training set : 10% randomly selected to be in development set
indexes = np.random.randint(0, len(data), int(len(data) * 0.1))
dev = [data[i] for i in indexes]
train = [s for s in data if s not in dev]

with open('../data/train.txt', 'w') as filehandle:
    for listitem in train:
        filehandle.write('%s\n' % listitem)

with open('../data/development.txt', 'w') as filehandle:
    for listitem in dev:
        filehandle.write('%s\n' % listitem)


#Creating a label file which consists of the different classes/labels which are present in the dataset
label_set = set([])
for item in data:
    pattern = re.compile(r'\w+:\w+\s')
    label = pattern.search(item).group().strip()
    label_set.add(label)
labels = list(label_set)
labels.sort()

with open('../data/label.txt', 'w') as filehandle:
    for listitem in labels:
        filehandle.write('%s\n' % listitem)

#Creating a vocabulary file with the help of stop_words file and the dataset
stop_words_file= open('../data/stopwords.txt','r')
stop_words=stop_words_file.read().split()

#Extracting the questions from each sentence by removing the labels
questions = [re.sub(r'\w+:\w+\s', '', s).lower() for s in data]

words = ' '.join(questions).split()

#Words which are part of stopwords file but good to keep in the vocabulary since it is related to the dataset
keepstr = ['what','which','who','whom','when','where','why','how']

vocabulary = {}
for word in words:
    if (word in stop_words) & (word not in keepstr):
        continue
    if word in vocabulary.keys():
        vocabulary[word] += 1
    else:
        vocabulary[word] = 1

vocabulary = dict(sorted(vocabulary.items(), key=lambda x: x[1], reverse=True))

word_count = sum(vocabulary.values())
vocabs_str = [("%s %d" % (key, value)) for key, value in vocabulary.items()]

with open('../data/vocabulary.txt', 'w') as filehandle:
    for listitem in vocabs_str:
        filehandle.write('%s\n' % listitem)
