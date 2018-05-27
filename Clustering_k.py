# opening file and splitting it

with open("C:/Users/osies/Desktop/PythonPractice/Pluralsight/machine-learning-algorithms/sentiment labelled sentences/sentiment labelled sentences/imdb_labelled.txt", "r") as text_file:
    lines = text_file.read().split('\n')

# splitting lines
lines = [line.split('\t') for line in lines if len(line.split('\t')) == 2 and line.split('\t')[1] != '']

# first element in each row
train_documents = [line[0] for line in lines]


from sklearn.feature_extraction.text import TfidfVectorizer

# TFID word frequency multiplied by the inverse document frequency
# max_df and min_df are parameters
# stop words removes words like the, a, an etc just verify the language

tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')

# fit_transform converts our document into tfid form
train_documents = tfidf_vectorizer.fit_transform(train_documents)

# Clusterin algorithm
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1, verbose=True)
kms = km.fit(train_documents)

print(kms)

# loop for printing according to the label
count = 0
for i in range (len(lines)):
    if count > 3:
        break
    if km.labels_[i] == 2:
        print(lines[i])
        count += 1


