import implicit
import pandas as pd

# import file
dataFile = 'C:/Users/osies/Desktop/PythonPractice/Pluralsight/machine-learning-algorithms/ml-100k/u.data'
# use column 0,1,2
data = pd.read_csv(dataFile, sep="\t", header=None, usecols=[0,1,2],names=['useId', 'itemId', 'rating'])
# check the table
print(data.head())

from scipy.sparse import coo_matrix

# converts it into a matrix
data['userId'] = data['userId'].astype("category")
data['itemId'] = data['itemId'].astype("category")
rating_matrix = coo_matrix((data['rating'].astype(float),
                       (data['itemId'].cat.codes.copy(),
                        data['userId'].cat.codes.copy())))

# calling least squares method through implicit module
user_factors, item_factors = implicit.alternating_least_squares(rating_matrix, factors=10, regularization=0.01)

user196=item_factors.dot(user_factors[196])

import heapq
# sort into descending and pick the top 3 users
heapq.nlargest(3, range(len(user196)), user196.take)