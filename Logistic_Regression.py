import numpy as np
import matplotlib as plt
import pandas as pd

# Importing Data and dealing with missing data
from sklearn.impute import SimpleImputer
missing_value_formats = ["n.a.","?","NA","n/a", "na", "--"]
dataset = pd.read_csv('breast-cancer-wisconsin.csv', na_values=missing_value_formats)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Replacing missing data with mean value of column 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X)
X = imputer.transform(X)

# Splitting data into training and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Training Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Testing One Prediction 
# print(classifier.predict([[5,1,1,1,2,1,3,1,1]]))

# Predicting the Test Set Results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_pred))
print()
print('Accuracy: {:.2f} %'.format((82+51)/(82+51+3+4)))
print()

# k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y = y_train, cv=10)
print('Accuracy: {:.2f} %'.format(accuracies.mean()*100))
print('Standard Deviation: {:.2f} %'.format(accuracies.std()*100))



