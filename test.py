from __future__ import division
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


modelName="finalMark.pkl"
#LOADING DATA
student_data = pd.read_csv("dataset.csv")
print "Student data loaded successfully!"


# Splitting to X and Y
feature_cols = list(student_data.columns[:-2])
target_col = student_data.columns[-2]
X_all = student_data[feature_cols]
y_all = student_data[target_col]



def preprocess_features(X):
    output = pd.DataFrame(index = X.index)
    for col, col_data in X.iteritems():
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)  
        output = output.join(col_data)
    return output
X_all = preprocess_features(X_all)
vvv=X_all.values.tolist()


print 
print
idOfStudent=int(raw_input('Enter the Student ID for Prediction  :'))-1



clf = joblib.load("linearRegression.pkl")
#print vvv[idOfStudent]
predictedLR=clf.predict([vvv[idOfStudent]])

clf1 = joblib.load("knn.pkl")
#print vvv[idOfStudent]
predictedknn=clf1.predict([vvv[idOfStudent]])

clf2 = joblib.load("dt.pkl")
#print vvv[idOfStudent]
predicteddt=clf2.predict([vvv[idOfStudent]])


 
print "Actual Mark is :",y_all[idOfStudent]
print("")
print("----------------------------------------------------------------------------------------")
print("ALGORITHM				|	  PREDICTED MARK	       ")
print("----------------------------------------------------------------------------------------")
print("Linear Regression                       |          ""%.2f" % predictedLR[0])   		
print("K-Nearest Neighbour                     |          ""%.2f" % predictedknn[0])
print("Decision Tree	                        |          ""%.2f" % predicteddt[0])                 
print("----------------------------------------------------------------------------------------")


