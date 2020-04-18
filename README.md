# QuestionClassifier

This code is an implementation of two different models, Bag-of-words and BiLSTM, to build a question classifier. The goal is to achieve an optimal accuracy when running classification on the test dataset, consisting sentence questions with pre-labelled class, by improving the models’ architecture and finding the best hyperparameters for this task. It is deduced that the utilisation of pre-trained embedded vectors produce better results on small datasets. The size of the dataset has also been found to affect the performance of each models.

The goal of question classification is to assign the input with appropriate categories associated with their corresponding answers respectively. The purpose of this experiment is to determine the labels of the given questions using two models, BOW and BiLSTM, subsequently calculating the optimal hyperparameters for the best performance on the test dataset. 

#
The dataset and stop word file must be kept in the data folder and named : dataset.txt; stop_words.txt

Config file:
Give the path for the following files:
1. Entire dataset,
2. Stop words list
3. Train Set
4. Development Set
5. Test Set
6. Label list
7. Vocab list


Preprocessing step:
Creates a train dev split, list of labels, vocabulary of words used in the dataset

Run the file:
python preprocess.py

-train and -test can be used together also.
-config is used to enter the path for the config file. 
By default it is hardcoded as: ./bow.config.yaml

For Training:
python question_classifier.py -train -config ./bilstm.config.yaml

Testing:
python question_classifier.py -test -config ./bilstm.config.yaml
