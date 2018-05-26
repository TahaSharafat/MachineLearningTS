import pandas as pd
import numpy as np

def readFile(filename, tbond=False):
    data = pd.read_csv(filename, sep=",", usecols=[0, 6], names=['Date', 'Price'], header=0)
    if tbond == False:
        returns = np.array(data["Price"][:-1], np.float)/np.array(data["Price"][1:], np.float)-1
        data["Returns"] = np.append(returns, np.nan)
    if tbond == True:
        data["Returns"] = data["Price"]/100
    data.index = data["Date"]
    data = data["Returns"][0:-1]
    return data

googData = pd.read_csv(r"C:\Users\osies\Desktop\PythonPractice\Pluralsight\machine-learning-algorithms\Regression\goog.csv")
nasdaqData = pd.read_csv(r"C:\Users\osies\Desktop\PythonPractice\Pluralsight\machine-learning-algorithms\Regression\nasdaq.csv")
tbondData = pd.read_csv(r"C:\Users\osies\Desktop\PythonPractice\Pluralsight\machine-learning-algorithms\Regression\tbonds5year.csv")

from sklearn.linear_model import SGDRegressor, LinearRegression

reg = SGDRegressor(eta0=0.1, n_iter=100000, fit_intercept=False)

reg.fit((nasdaqData-tbondData).reshape(-1, 1), (googData-tbondData))
# reg.coef_

