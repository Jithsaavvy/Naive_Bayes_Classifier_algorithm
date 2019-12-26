# Naive_Bayesian_Classifier---Bayes-Rule

The Naive Bayes classifier is a quite simple and popular classifier that is entirely based on a conditional independence assumption

In this exercise, I implement my own Naive Bayes classifier that can be used for predicting the stability of object placements on a table. The scenario is one in which our robot Jenny is putting objects on a table, such that we'll suppose that the robot chooses a random continuous table pose for placement and then tries to predict whether placing a point object there will be successful by describing the pose with a few features.

Let's suppose that a pose is described using the following features, all of which are discrete:
1. Inside table: Takes the values 0 and 1, corresponding to whether a pose is outside or inside the table respectively.
2. Distance to the robot: Takes the values 0, 1, and 2, corresponding to very close, reachable, and far.
3. Minimum distance to the other objects: Takes the values 0, 1, and 2, corresponding to very close, close, and far.
4. Distance to the closest surface edge: Also takes the values 0, 1, and 2, corresponding to very close, close, and far.

Each pose either leads to a successful execution or not, so we have two classes, namely 00 and 11, corresponding to the outcomes failure and success respectively.


The task consists of the following steps:

1. Training data set (data/train.txt) of features describing 1500 poses and the class labels of these. Use the data in this data set for learning the prior probabilities P(Cj)and the conditional probabilities P(Fi|Cj), i∈{1,2,3,4}, j∈{1,2}j∈{1,2} . Learning in this context means calculating the values of the probabilities as relative frequencies.

2. The test data set (data/test.txt) for testing the classifier (i.e. predict the class labels of the 500 test points using the given features and compare the predicted labels with the actual labels) along with confusion matrix

Note:
The green and black points correspond to stable and unstable placements respectively
