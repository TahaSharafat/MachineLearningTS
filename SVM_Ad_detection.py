import numpy as np
import pandas as pd

dataFile = 'C:/Users/osies/Desktop/PythonPractice/Pluralsight/machine-learning-algorithms/ad-dataset/ad.data'
data = pd.read_csv(dataFile, sep=",", header=None, low_memory=False)

#print(data.head(20))

# Check whether a given value is missing, # if so then convert to NaN

def toNum(cell):
    try:
        return np.float(cell)
    except:
        return np.nan

# apply missing value check to a column / Panda series

def seriestoNum(series):
    return series.apply(toNum)

# train data

train_data = data.iloc[0:,0:-1].apply(seriestoNum)
# print(train_data.head(20))

# subset to those rows that has no missing values:

train_data = train_data.dropna()
# print(train_data.head(20))

# function for the last column of the
# table to return 1 if ad or else 0 if not ad

def toLabel(str):
    if str == "ad.":
        return " is ad "
    else:
        return " not a ad "

# train label for the last column
# which shows if its ad or not

train_labels = data.iloc[train_data.index,-1].apply(toLabel)

# print(train_labels)

# Training phase with Support Vector Machine

from sklearn.svm import LinearSVC

# use fit methiod which can be used for training phase

clf = LinearSVC()
fittest = clf.fit(train_data[100:2300], train_labels[100:2300])

#print(fittest)

# Test phase to see the prediction

testingphase = clf.predict(train_data.iloc[12].values.reshape(1, -1))

print(testingphase)