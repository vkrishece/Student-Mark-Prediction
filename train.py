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
print "Student data loaded successfully!\n\n"


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




#Shuffle and Split with Equal Graduation rate
X_all, y_all = shuffle(X_all, y_all, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,test_size=0.15, random_state=42)
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])
print "Data Shuffle and Split is Done."
print "\n\n\n------------------"




def metrices(y_true, y_pred):
	from sklearn.metrics import explained_variance_score
	explained_variance_score=explained_variance_score(y_true, y_pred)  #http://scikit-learn.org/stable/modules/model_evaluation.html

	from sklearn.metrics import mean_absolute_error
	mean_absolute_error=mean_absolute_error(y_true, y_pred)

	from sklearn.metrics import mean_squared_error
	mean_squared_error=mean_squared_error(y_true, y_pred)

	#from sklearn.metrics import mean_squared_log_error
	#mean_squared_log_error=mean_squared_log_error(y_true, y_pred)  

	from sklearn.metrics import median_absolute_error
	median_absolute_error=median_absolute_error(y_true, y_pred)

	from sklearn.metrics import r2_score
	r2_score=r2_score(y_true, y_pred) 
	print "         Statistics           "
	print "++++++++++++++++++++++++++++++++++++++++++++" 
	print "Variance	     :",explained_variance_score
	print "Mean Absolute Error  :",mean_absolute_error
	print "Mean Squared Error   :",mean_squared_error
	print "Median Absolute Error:",median_absolute_error
	print "R2 Score             :",r2_score
	print "++++++++++++++++++++++++++++++++++++++++++++" 
	#return explained_variance_score,mean_absolute_error,mean_squared_error,median_absolute_error,r2_score



def model(X_train, X_test, y_train, y_test):
	print
	print
	print "Linear Regression Model:"
	print "-----------------------"
	lm = linear_model.LinearRegression()
	model = lm.fit(X_train, y_train)
	y_pred = lm.predict(X_test)
	#print "MODEL2"
	#print(y_pred)
	#print "Model Score",lm.score(X_train, y_train)
	#print "Model Co-efficients",lm.coef_
	#print "Model Intercepts",lm.intercept_
	metrices(y_test, y_pred)
	joblib.dump(model, "linearRegression.pkl") 
	print "Model Created for Linear Regression"


def model2(X_train, X_test, y_train, y_test):
	print
	print
	print "KNN Model"
	print "---------"
	from sklearn import neighbors
	n_neighbors = 5
	for i, weights in enumerate(['uniform', 'distance']):
	    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
	    knn.fit(X_train, y_train)
	    y_pred=knn.predict(X_test)
	metrices(y_test, y_pred)
	joblib.dump(knn, "knn.pkl") 
	print "Model Created for KNN"


def model3(X_train, X_test, y_train, y_test):
	print
	print
	print "Decision Tree Model"
	print "-------------------"
	from sklearn import tree
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred=clf.predict(X_test)
	metrices(y_test, y_pred)
	joblib.dump(clf, "dt.pkl") 
	print "Model Created for Decision Tree"
	

model(X_train, X_test, y_train, y_test)
model2(X_train, X_test, y_train, y_test)
model3(X_train, X_test, y_train, y_test)
print "Training is Done with 3 models..."

