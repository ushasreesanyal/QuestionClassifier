# QuestionClassifier

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
