from sklearn import datasets
import sklearn
from sklearn import model_selection, dummy, ensemble, linear_model, neural_network
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
### load dataset
boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
### split into test and training sets
X = df
y = pd.DataFrame(boston.target, columns=['MEDVAL'])
df = X.join([y])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split (X,
                                                     y,
                                                     test_size=0.2,
                                                     random_state=0)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
#make a dummy model
model_dummy = sklearn.dummy.DummyRegressor(strategy = 'median')
model_dummy.fit(X_train, y_train)
print("dummy score:" + str((model_dummy.score(X_test, y_test))))
#dummy score:-0.008249660981729745

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


#kitchen sink random forest
kitchen_sink_rf_model = sklearn.ensemble.RandomForestRegressor(random_state=0, n_estimators=100)
kitchen_sink_rf_model.fit(X_train, y_train)
print("kitchen sink random forest regression: " + str(kitchen_sink_rf_model.score(X_test, y_test)))
#kitchen sink random forest regression: 0.7715156607153403

#kitchen sink random forest scaled
kitchen_sink_rf_scaled_model = sklearn.ensemble.RandomForestRegressor(random_state=0, n_estimators = 100)
kitchen_sink_rf_scaled_model.fit(X_train_scaled, y_train)
print("kitchen sink random forest scaled regression: " + str(kitchen_sink_rf_scaled_model.score(X_test_scaled, y_test)))
#kitchen sink random forest scaled regression: 0.7420904474375429

#interesting, the scaled random forrest performs worse than unscaled

#Linear Regression
linear_regression_model = sklearn.linear_model.LinearRegression()
linear_regression_model_scaled = sklearn.linear_model.LinearRegression()
linear_regression_model.fit(X_train, y_train)
linear_regression_model_scaled.fit(X_train_scaled, y_train)
print("linear regression model: " +
      str(linear_regression_model.score(X_test, y_test)))
#linear regression model: 0.5892223849182507
print("linear regression model scaled: " +
      str(linear_regression_model_scaled.score(X_test_scaled, y_test)))
#linear regression model scaled: 0.5355752681923033

#Neural Network
nn_model = sklearn.neural_network.MLPRegressor(max_iter=2000)
nn_scaled_model = sklearn.neural_network.MLPRegressor(max_iter=2000)
nn_model.fit(X_train, y_train)
nn_scaled_model.fit(X_train_scaled, y_train)
print("kitchen sink neural network unscaled: " + str(nn_model.score(X_test, y_test)))
#kitchen sink neural network unscaled: 0.32364156937942623
print("kitchen sink neural network scaled: " + str(nn_scaled_model.score(X_test_scaled, y_test)))
#kitchen sink neural network scaled: 0.5400981194338121

#neural network performs worse than random forest. Lets improve random forest

#list importances

#rename model


#get importances
importances = list(kitchen_sink_rf_model.feature_importances_)
#list tuples with variable and importance
feature_list = df.columns
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#for feature in feature_importances:
#    print ('Feature: {f}, Importance: {i}'.format(f=feature[0], i=feature[1]))

#Feature: RM, Importance: 0.42
#Feature: LSTAT, Importance: 0.41
#Feature: CRIM, Importance: 0.04
#Feature: DIS, Importance: 0.04
#Feature: NOX, Importance: 0.02
#Feature: TAX, Importance: 0.02
#Feature: PTRATIO, Importance: 0.02
#Feature: INDUS, Importance: 0.01
#Feature: AGE, Importance: 0.01
#Feature: RAD, Importance: 0.01
#Feature: B, Importance: 0.01
#Feature: ZN, Importance: 0.0
#Feature: CHAS, Importance: 0.0

sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
cumulative_importances = np.cumsum(sorted_importances)
#assign each variable an x location for plotting
x_values = list(range(len(importances)))
plt.plot(x_values, cumulative_importances, 'g-')

#draw 95% importance cutoff
plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color='r', linestyles='dashed')
#lables and ticks
plt.xticks(x_values, sorted_features, rotation = 'vertical')
plt.title('Cumulative Importances')
plt.show()
#seems like we can omit some. ZN, CHAS) 
important_feature_names = [feature[0] for feature in feature_importances[0:-2]]
X_train2 = X_train[important_feature_names]
X_test2 = X_test[important_feature_names]

rf_model_improved = sklearn.ensemble.RandomForestRegressor(random_state=0, n_estimators=100)
rf_model_improved.fit(X_train2, y_train)
print("improved random forest: " + str(rf_model_improved.score(X_test2, y_test)))

#improved random forest: 0.7748074713295194
#seems like the random forrest is doing a good job reducing features on its own.
#lets go back to kitchen sink model
plt.xticks(ticks=None, labels=None)
y_pred = kitchen_sink_rf_model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.grid()
plt.show()

