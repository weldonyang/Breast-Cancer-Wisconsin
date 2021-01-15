import numpy as np
import matplotlib as plt
import pandas as pd

dataset = pd.read_csv('breast-cancer-wisconsin.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values='?', strategy='mean')
X = imputer.fit_transform(X)
print(X)

# Splitting data into training and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

print(classifier.predict([[5,1,1,1,2,1,3,1,1]]))



