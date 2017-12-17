'''

This algorithm is trained and tested on a breast cancer dataset, so as to classify the cancer as belign or malignant

'''

import numpy as np
from sklearn import neighbors, preprocessing, cross_validation
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.csv')
df.replace('?', 99999, inplace=True)
df.drop(['id'], 1, inplace = True)

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print("Accuracy: " + str(accuracy))

# Classifying a random data point
example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1) #because predicting 1D arrays is deprecated
prediction = clf.predict(example_measures)
print("Belign" if prediction == 2 else "Malignant")