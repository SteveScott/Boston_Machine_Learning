from sklearn import datasets
import sklearn
from sklearn import model_selection, dummy, ensemble, linear_model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
### load dataset
boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
### split into test and training sets
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split (X,
                                                     y,
                                                     test_size=0.2,
                                                     random_state=0)

#make a dummy model
model_dummy = sklearn.dummy.DummyRegressor(strategy = 'median')
model_dummy.fit(X_train, y_train)
print("dummy score:" + str((model_dummy.score(X_test, y_test))))
#dummy score:-0.031141019169146356

#kitchen sink random forest
kitchen_sink_model = sklearn.ensemble.RandomForestRegressor(random_state=0)
kitchen_sink_model.fit(X_train, y_train)
print("kitchen sink random forest regression: " + str(kitchen_sink_model.score(X_test, y_test)))
kitchen sink random forest regression: 0.6382757610422984

linear_regression_model = sklearn.linear_model.LinearRegression()
linear_regression_model.fit(X_train, y_train)
print("linear regression model: " +
      str(linear_regression_model.score(X_train, y_train)))
#linear regression model: 0.6830386021775365