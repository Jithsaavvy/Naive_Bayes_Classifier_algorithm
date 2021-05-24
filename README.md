# Naive_Bayes_Classifier_algorithm

The Naive Bayes classifier is a quite simple and popular classifier that is entirely based on a conditional independence assumption

In this exercise, I implemented my own **Naive Bayes classifier** that can be used for predicting the stability of object placements on a table.  Statistical measures such as **classification error, accuracy, precision, recall values, confidence interval** are also determined for the ML classifier model developed. Thus Classifier performance is reported. 

The scenario is one in which a robot is putting objects on a table, such that the robot chooses a random continuous table pose for placement and then tries to predict whether placing a point object there will be successful by describing the pose with a few features.

The task consists of the following steps:

1. Training data set consisting of features about 1500 poses along with class labels.

2. The test data set which is used for testing the classifier (i.e. The class labels of the 500 test data points are tested using the given features and the predicted labels are compared with the actual labels) and 

3. The performance evaluation/statistics measure of the classifier is done as follows by means of confusion matrix <br>
      A) **Classification Accuracy and Error** are calculated <br>
      B) **Precision** and **Recall** values are also calculated as part of performance evaluation. <br> 
      C) **Confidence Interval for the classification error** is determined and evaluated with the testing datasets. <br>

4. **99.6%** accuracy is achieved

To assess the performance, **Evaluation datasets** are used for testing the algorithm
