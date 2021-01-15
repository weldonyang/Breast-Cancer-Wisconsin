import numpy as np
import matplotlib as plt
import pandas as pd

dataset = pd.read_csv('breast-cancer-wisconsin.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values