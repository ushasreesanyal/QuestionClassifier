#!/usr/bin/env python
import torch
import numpy as np
import random
import re
import argparse
import yaml
import sys

#Make vector for question and label
def create_sentence_embedding(sentence,labels,vocabulary,stop_words):
        label, question = sentence
        label = torch.LongTensor([labels.index(label)]) #extract question index and make label embedding
        embedding = []
        embedding = get_word_index(question,vocabulary,stop_words) #extract question index and make question embedding
        embedding = torch.LongTensor(embedding).unsqueeze(-2)
        return label, embedding

#Convert each question to index and ignore if it is present in the stopwords
def get_word_index(question,vocabulary,stop_words):
    question = question.lower()
    indexes = []
    keepstr = ['what','which','who','whom','when','where','why','how']
    for word in question.split():
        if ((word in stop_words) & (word not in keepstr)):
            continue
        if word in vocabulary:
            indexes.append(vocabulary.index(word))
        else:
            indexes.append(vocabulary.index('#unk#'))
    return indexes

#Load the dataset and assigns value for dataset variable
def load_dataset(dataset,path):
    with open(path, 'r') as dataset_file:
        for line in dataset_file:
            sample = line.split(' ', 1)
            label = sample[0]
            question = sample[1]
            dataset.append((label, question))
    return dataset

#Assigns value to the label by reading from the file
def load_labels(path):
    with open(path, 'r') as labels_file:
        labels = labels_file.read().split('\n')
    return labels

#Loads the vocabulary and if pre train file is present; uses the words from the file to
#assign a weighted value which will be used to create the word vector
def load_vocabulary(path,path_pre_emb,pre_train_words):
    k = 3
    with open(path, 'r') as vocabulary_file:
        vocabulary.append('#unk#')
        if path_pre_emb == "None":
            for line in vocabulary_file:
                pair = line.split()
                if int(pair[1]) > k:
                    vocabulary.append(pair[0])
        else:
            pre_train_words = load_pre_train(path_pre_emb,pre_train_words)
            pre_weight.append(np.random.rand(300))
            for line in vocabulary_file:
                pair = line.split()
                if int(pair[1]) > k and pair[0] in pre_train_words.keys():  # k = 3
                    vocabulary.append(pair[0])
                    pre_weight.append(pre_train_words[pair[0]])
    return pre_weight,vocabulary

#Extracts the pre trained weight and creates key value pair
def load_pre_train( path,pre_train_words):
    with open(path, 'r') as pre_train_file:
        for line in pre_train_file:
            line = re.sub('\s+',' ',line).rstrip()
            pair = line.split(' ')
            key = pair[0]
            value = [float(x) for x in pair[1:]]
            pre_train_words[key] = value
        return pre_train_words

def load_stop_words( path):
    with open(path, 'r') as stop_words_file:
        stop_words = stop_words_file.read().split()
        return stop_words


class BOWClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, label_size, pre_train_weight=None, freeze=False):
        super(BOWClassifier, self).__init__()
        if pre_train_weight is None:
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.embeddingBag = torch.nn.EmbeddingBag(vocab_size, embedding_dim)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(pre_train_weight, freeze=freeze)
            self.embeddingBag = torch.nn.EmbeddingBag.from_pretrained(pre_train_weight, freeze=freeze)

        self.fc1 = torch.nn.Linear(embedding_dim, 128)
        self.fc2 = torch.nn.Linear(128, label_size)

    def forward(self, x):
        out = self.embeddingBag(x)
        out = self.fc1(out)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        return out

class BiLSTMClassifier(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim, label_size, pre_train_weight=None, freeze=False):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_state_size = int(embedding_dim / 2)
        if pre_train_weight is None:
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.embeddingBag = torch.nn.EmbeddingBag(vocab_size, embedding_dim)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(pre_train_weight, freeze=freeze)
            self.embeddingBag = torch.nn.EmbeddingBag.from_pretrained(pre_train_weight, freeze=freeze)

        self.bilstm = torch.nn.LSTM(embedding_dim, self.hidden_state_size, bidirectional=True)
        self.linear = torch.nn.Linear(embedding_dim, label_size)

    def forward(self, x):
        embeds = self.embedding(x)
        seq_len = len(x[0])
        bilitm_out, _ = self.bilstm(embeds.view(seq_len, 1, -1))
        out = torch.cat((bilitm_out[0, 0, self.hidden_state_size:],
                         bilitm_out[seq_len - 1, 0, :self.hidden_state_size])).view(1, -1)
        return torch.nn.functional.log_softmax(self.linear(out))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS,description='Question Classifier using Pytorch')
    parser.add_argument('-config', default='./bow.config.yaml',help='Configuration File Path')
    parser.add_argument('-train', action='store_true',default=argparse.SUPPRESS,help='Training phase')
    parser.add_argument('-test',action='store_true',default=argparse.SUPPRESS, help='Testing phase')

    args = parser.parse_args()


    # load config
    with open(args.config, 'r') as ymlfile:
        conf = yaml.safe_load(ymlfile)

    dataset = []
    labels = []
    vocabulary = []
    stop_words = []
    pre_weight = []
    pre_train_words = {}
    try:
        labels = load_labels(conf['path_labels'])
        pre_weight, vocabulary = load_vocabulary(conf['path_vocab'],conf['path_pre_emb'],pre_train_words)
        stop_words = load_stop_words(conf['path_stop_words'])
    except FileNotFoundError:
        print ("File not found. Please check config file.")

    if (conf['path_pre_emb'] == "None"):
        pre_train_weight = None
    else:
        pre_train_weight=torch.FloatTensor(pre_weight)

    print("Model:",conf['model'])
    if("train" in args):
        print("Training Phase")
        #Training Phase
        if(conf['model'].lower()=='bow'):
            model = BOWClassifier(len(vocabulary), conf['word_embedding_dim'], len(labels),
            pre_train_weight=pre_train_weight,freeze=conf['freeze'])

        if(conf['model'].lower()=='bilstm'):
            model = BiLSTMClassifier(len(vocabulary), conf['word_embedding_dim'], len(labels),
            pre_train_weight=pre_train_weight,freeze=conf['freeze'])

        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),conf['lr_param'])

        train_dataset = []
        try:
            train_dataset = load_dataset(train_dataset,conf['path_train'])
        except FileNotFoundError:
            print ("Model path file not found. Please check config file.")

        for e in range(0, conf['epoch']):
            error = 0
            for sentence in train_dataset:
                target, question = create_sentence_embedding(sentence,labels,vocabulary,stop_words)
                optimizer.zero_grad()
                target_pred = model(question)
                loss = loss_function(target_pred, target)
                error += loss.item()
                loss.backward()
                optimizer.step()
            print('%d epoch finish, loss: %f' % (e + 1, error / train_dataset.__len__()))

        try:
            torch.save(model, conf['path_model'])
        except FileNotFoundError:
            print ("Model path file not found. Please check config file.")

        #Development Phase - Add Hyperparameter tuning
        dev_dataset = []
        try:
            dev_dataset = load_dataset(dev_dataset,conf['path_dev'])
        except FileNotFoundError:
            print ("Model path file not found. Please check config file.")

        acc = 0
        for sentence in dev_dataset:
            target, test = create_sentence_embedding(sentence,labels,vocabulary,stop_words)
            output = model(test)
            _, pred = torch.max(output.data, 1)
            if target == pred:
                acc += 1
        print('Dev set acc: ' + str(acc))
        acc_rate = float(acc) / float(dev_dataset.__len__())
        print('Dev set acc_rate: ' + str(acc_rate))

    if("test" in args):
        print("Testing Phase")
        #Testing Phase
        try:
            model= torch.load(conf['path_model'])
            test_dataset = []
            try:
                test_dataset = load_dataset(test_dataset,conf['path_test'])
            except FileNotFoundError:
                print ("Model path file not found. Please check config file.")

            outputStr = []
            acc = 0
            for sentence in test_dataset:
                target, test = create_sentence_embedding(sentence,labels,vocabulary,stop_words)
                output = model(test)
                _, pred = torch.max(output.data, 1)
                outputStr.append('Target: %s, Predicted: %s ' % (labels[target.item()], labels[pred.item()]))
                if target == pred:
                    acc += 1
            acc_rate = float(acc) / float(test_dataset.__len__())
            outputStr.append('Accuracy: ' + str(acc_rate*100)+'%')
            with open(conf['path_eval_result'], 'w') as filehandle:
                for listitem in outputStr:
                    filehandle.write('%s\n' % listitem)
        except FileNotFoundError:
             print ("File not found. Please check config file.")
