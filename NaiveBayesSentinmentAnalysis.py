# opening file and splitting it

with open("C:/Users/osies/Desktop/PythonPractice/Pluralsight/machine-learning-algorithms/sentiment labelled sentences/sentiment labelled sentences/imdb_labelled.txt", "r") as text_file:
    lines = text_file.read().split('\n')

# appending different txt in one file
with open("C:/Users/osies/Desktop/PythonPractice/Pluralsight/machine-learning-algorithms/sentiment labelled sentences/sentiment labelled sentences/yelp_labelled.txt", "r") as text_file:
    lines += text_file.read().split('\n')

with open("C:/Users/osies/Desktop/PythonPractice/Pluralsight/machine-learning-algorithms/sentiment labelled sentences/sentiment labelled sentences/amazon_cells_labelled.txt", "r") as text_file:
    lines += text_file.read().split('\n')

# splitting lines
lines = [line.split('\t') for line in lines if len(line.split('\t')) == 2 and line.split('\t')[1] != '']

# training data set in a separate line 0 for words
train_documents = [line[0] for line in lines]


# training data set in a separate line 1 for labels
train_labels = [line[1] for line in lines]

# print(train_labels)

# use scikit learn to vectorize

from sklearn.feature_extraction.text import CountVectorizer

# initiate counter vectorizer in sklearn
count_vectorizer = CountVectorizer(binary='true')

# use counter vectorizer fit transform to convert the training document
# into tuples of number which represents the frequency of words

train_documents = count_vectorizer.fit_transform(train_documents)

# print(train_documents) # will print only elements that are 1 and will leave all the 0's

# Training phase

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB().fit(train_documents, train_labels)

# Testing phase

testing = classifier.predict(count_vectorizer.transform(["this is the worst movie"]))

print(testing)