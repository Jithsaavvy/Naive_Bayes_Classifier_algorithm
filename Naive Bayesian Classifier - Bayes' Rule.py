import numpy as np
import matplotlib.pyplot as plt

'''
Implement my own Naive Bayes classifier that can be used for predicting the stability of object placements on a table.
The pose is described using the following features.
Each pose either leads to a successful execution or not, so we have two classes, namely 0 and 1, corresponding to the outcomes failure and success respectively
'''

#Loading training data
train_data = np.loadtxt("data/train.txt")

#Loading testing data
test_data = np.loadtxt("data/test.txt")

#Loading given points for testing
test_points = np.loadtxt("data/test_points.txt")

#Dictionaries to store features of class 0 and 1
features_class0 = {i:{} for i in range(0, 4)}
features_class1 = {i:{} for i in range(0, 4)}
predicted_labels = []
features = 4
class_data = [0,1] #Since class contains only 0 and 1 from input data
class_len = []   #Stores length of class data C
true_pos,false_pos,true_neg,false_neg=0,0,0,0
class_prob = []

for i in range(len(class_data)):
    det = [x for x in train_data[:,4] if x==class_data[i]]  #Stores the class data 0 and 1
    class_len.append(len(det))
    class_prob.append(len(det)/len(train_data[:,-1]))

#Extracting feature data F1 , F2, F3, F4 and classifying into class o or 1    
for i in range(features):
    for j in range(3):
        class_0 = [x for x,y in train_data[:,[i,4]] if x==j and y==0]
        class_1 = [x for x,y in train_data[:,[i,4]] if x==j and y==1]
        features_class0[i][j] = len(class_0)/class_len[0]
        features_class1[i][j] = len(class_1)/class_len[1]

for i in range(len(test_data)):
    prob_features_givenclass_0 = class_prob[0]*features_class0[0][test_data[i,0]]*features_class0[1][test_data[i,1]]*features_class0[2][test_data[i,2]]*features_class0[3][test_data[i,3]]
    prob_features_givenclass_1 = class_prob[1]*features_class1[0][test_data[i,0]]*features_class1[1][test_data[i,1]]*features_class1[2][test_data[i,2]]*features_class1[3][test_data[i,3]]
    
    #Choosing K=max(p...) as the class label
    if(prob_features_givenclass_1 > prob_features_givenclass_0):
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)
        
    #For Confusion Matrix Values
    if(test_data[i,-1] == 0 and predicted_labels[i] == 0):     #True Negative
        true_neg=true_neg+1
      
    elif(test_data[i,-1] == 0 and predicted_labels[i] == 1):   #False Positive
        false_pos=false_pos+1
        
    if(test_data[i,-1] == 1 and predicted_labels[i] == 1):     #True Positive
        true_pos=true_pos+1
        
    elif(test_data[i,-1] == 1 and predicted_labels[i] == 0):   #False Negative
        false_neg=false_neg+1

# Plot your predicted labels.
for i in range(len(test_data)):
    if predicted_labels[i]==0:      #Plot labels 0 as black
        plt.plot(test_points[i,0],test_points[i,1],color="black", marker='o')
    else:       #Plot labels 1 as green
        plt.plot(test_points[i,0],test_points[i,1],color="green", marker='o')
        
# Please assign the values of the confusion matrix to the following variables.
true_positive = true_pos
false_positive = false_pos
true_negative = true_neg
false_negative = false_neg

# Please assign the accuracy to the following variable.
accuracy = ((true_positive+true_negative)/len(test_data)*100)
print("Prediction Accuracy: ",accuracy,"%")
print("True Positive: ", true_positive)
print("False Positive: ",false_positive)
print("True Negative: ", true_negative)
print("False Negative: ", false_negative)
