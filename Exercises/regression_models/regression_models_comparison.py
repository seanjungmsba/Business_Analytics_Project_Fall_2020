############################################ Import Data ###############################################
# To write a Python 2/3 compatible codebase, the first step is to add this line to the top of each module
from __future__ import division, print_function, unicode_literals

import pandas as pd # library providing high-performance, easy-to-use data structures and data 
# analysis tools for the Python programming language

df = pd.read_csv(r'C:\Users\seanj\Desktop\Intro to Business Analytics\HW4\BA_HW4.csv')  # Read the data from the local file, no header
print(df.shape)                                       # Print num of rows and columns
df.columns = ['US', 'Source1', 'Source2', 'Source3', 
              'Source4', 'Source5', 'Source6', 'Source7', 'Source8', 
              'Source9', 'Source10', 'Source11', 'Source12', 'Source13',
             'Source14', 'Source15', 'Source16', 'Freq.', 'last_update_days_ago', '1st_update_days_ago', 'Web_order',
             'Gender=mal', 'Address_is_res', 'Purchase', 'Spending'] # Column Names
df.head()                                             # Perview first rows of data
df.describe()

# # Exploring the dataset

############################################ Data Visualization ###############################################

# Libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns

cols = ['Freq.', 'last_update_days_ago', '1st_update_days_ago','Spending'] # Select Numeric Attributes

sns.pairplot(df[cols], height=2.5)              # Plot pairwise relationships in a dataset
plt.tight_layout()                            # Tight_layout automatically adjusts subplot params 
                                              # so that the subplot(s) fits in to the figure area.
# plt.savefig('BA_HW4.png', dpi=300) # Saves the figure in our local disk
plt.show()                                    # Display figure

############################################ Correlations (NTK) ############################################

import numpy as np

cm = np.corrcoef(df[cols].values.T) # Return Pearson product-moment correlation coefficients

# sns.set(font_scale=1.5)
# Heatmap visualisation of pearson correlation coefficients
# Documentation https://seaborn.pydata.org/generated/seaborn.heatmap.html
hm = sns.heatmap(cm,                # Plot rectangular data as a color-encoded matrix
                 cbar=True,         # Whether to draw a colorbar.
                 annot=True,        # If True, write the data value in each cell.
                 square=True,       # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                 fmt='.2f',         # String formatting code to use when adding annotations.
                 annot_kws={'size': 15}, # Keyword arguments for ax.text when annot is True
                 yticklabels=cols,  # If True, plot the column names of the dataframe.
                 xticklabels=cols)

plt.tight_layout()
# plt.savefig('correlation_coefficient.png', dpi=300) # Saves the figure in our local disk
plt.show()

# # Estimating Linear Regression Model Coefficients

################################### Fit Simple Linear Regression Model ###################################

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.stats import norm
import pandas as pd

# A simple linear regression with only one feature
X = df[["Freq."]] # Attribute
y = df[["Spending"]] # Target Variable

# Min, Max, Mean, median, standard deviation
print('The minimum of the target variable (in $):', np.min(y))
print('The maximum of the target variable (in $):', np.max(y))
print('The mean of the target variable (in $):', np.mean(y))
print('The median of the target variable (in $):', np.median(y))
print('The standard deviation of the target variable (in $):', np.std(y))

# Plot the distribution of the target variable
# Adapted from: https://seaborn.pydata.org/generated/seaborn.distplot.html
fig, ax = plt.subplots()
sns.distplot(y, bins=20, color="g", ax=ax, axlabel="Distribution of Target Variable: Spending amount (in $)")
plt.show()

################################### Visualize Linear Regression Model ###################################

# Fit linear regression
slr = LinearRegression() # Linear Regression class
slr.fit(X, y)            # Fit model to the data

print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

def lin_regplot(X, y, model): # Define function that takes as input X, y and model
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70) # A scatter plot of y vs x. s : scalar or array_like, shape (n, ), optional
    plt.plot(X, model.predict(X), color='black', lw=2) #lw = line width   
    return

lin_regplot(X, y, slr)       # Call the above function
plt.xlabel('Number of transactions in last year at source catalog [Freq.]') # Set label for x axis
plt.ylabel('Amount spent by customer in test mailing ($) [Spending]')       # Set label for y axis

#plt.savefig('images/10.02.png', dpi=300)
plt.show()

################################### Fit a Linear Regression Model #######################################

from sklearn.model_selection import train_test_split # Split validation class

X = df.iloc[:, :-1].values # Use all features as attributes except last column
y = df[["Spending"]].values # Target Variable   # Set last column as target variable

X_train, X_test, y_train, y_test = train_test_split( # Split validation
    X, y, test_size=0.3, random_state=42)

slr2 = LinearRegression()    # Linear Regression class
slr2.fit(X_train, y_train)   # Fit Model to data
y_train_pred = slr2.predict(X_train) # Apply model to train data
y_test_pred = slr2.predict(X_test)   # Apply model to test data

print('Slope: %.3f', slr2.coef_)
print('Intercept: %.3f' % slr2.intercept_)

######################## Linear Regression Model - Evaluation Metrics ##############################

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# See all regression metrics here http://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
print('MSE train: %.3f, test: %.3f' % ( # mean_squared_error
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('MAE train: %.3f, test: %.3f' % ( # mean_absolute_error
        mean_absolute_error(y_train, y_train_pred),
        mean_absolute_error(y_test, y_test_pred)))

print('RMSE test: %.3f' % ( # root_mean_squared_error
        np.sqrt(mean_squared_error(y_test, y_test_pred))))

##################################### Optimize linear regression Example ##########################################
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold

# Define 'inner_cv' and 'outer_cv' for the cross-validation steps
inner_cv = KFold(n_splits=10, shuffle=True)
outer_cv = KFold(n_splits=10, shuffle=True)

# split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Here I am trying to optimize the two parameters: 'fit_intercept' and 'normalize'
params = {'fit_intercept':[True, False], 'normalize':[True, False]}

lin = LinearRegression() # Linear Regression class

lin_opt = GridSearchCV(lin, params, cv=10)
lin_opt = lin_opt.fit(x_train, y_train) # Fit model to the data

pred=lin_opt.predict(x_test) #make prediction on test set  
print('RMSE value is:', np.sqrt(mean_squared_error(y_test,pred)))
      
print("Parameter Tuning")
print("Non-nested Performance: ", lin_opt.best_score_)
print("Optimal Parameter: ", lin_opt.best_params_)    # Parameter setting that gave the best results on the hold out data.
print("Optimal Estimator: ", lin_opt.best_estimator_) # Estimator that was chosen by the search, i.e. estimator which gave highest score

# Outer CV
nested_score_lin_opt = cross_val_score(lin_opt, X=X, y=y, cv=outer_cv)
print("Nested CV Performance: ",nested_score_lin_opt.mean(), " +/- ", nested_score_lin_opt.std())

# # k-NN for Regression

##################################### kNN Regressor Example #########################################
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error 
from math import sqrt

# split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalize data with min max scaling
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

# 3NN regressor
knn_regressor = neighbors.KNeighborsRegressor(n_neighbors = 3)

# Fit and Evaluate Model
knn_regressor.fit(x_train, y_train)  #fit the model
pred=knn_regressor.predict(x_test) #make prediction on test set
error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse

print('RMSE value is:', error)

##################################### Optimize kNN Regressor Example ##########################################

from sklearn.model_selection import GridSearchCV

# Here, I am trying to optimize two parameters:'n_neighbors' AND 'weights' AND 'p'
params = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'weights':['uniform', 'distance'], 'p':[1,2]}

knn_regressor2 = neighbors.KNeighborsRegressor()

knn2_optk = GridSearchCV(knn_regressor2, params, cv=10)
knn2_optk = knn2_optk.fit(x_train,y_train)

pred2=knn2_optk.predict(x_test) #make prediction on test set
print(knn2_optk.best_params_)
print('RMSE value is:', sqrt(mean_squared_error(y_test,pred2)))

print("Parameter Tuning")
print("Non-nested Performance: ", knn2_optk.best_score_)
print("Optimal Parameter: ", knn2_optk.best_params_)    # Parameter setting that gave the best results on the hold out data.
print("Optimal Estimator: ", knn2_optk.best_estimator_) # Estimator that was chosen by the search, i.e. estimator which gave highest score

# Outer CV
nested_score_knn2_optk = cross_val_score(knn2_optk, X=X, y=y, cv=outer_cv)
print("Nested CV Performance: ",nested_score_knn2_optk.mean(), " +/- ", nested_score_knn2_optk.std())

# # Decision Tree Regressor

#################################### Regressor Tree - Numeric Prediction ###########################################

from sklearn.tree import DecisionTreeRegressor #Documentation available here http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
from sklearn.model_selection import cross_val_score

# Assign attribute and target variable
X = df[["Freq."]].values # Attribute
y = df[["Spending"]].values # Target Variable

# split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regressor
# Supported criteria are “mse” for the mean squared error, which is equal to variance 
# reduction as feature selection criterion and minimizes the L2 loss using the mean of 
# each terminal node, “friedman_mse”, which uses mean squared error with Friedman’s 
# improvement score for potential splits, and “mae” for the mean absolute error, 
# which minimizes the L1 loss using the median of each terminal node.
tree = DecisionTreeRegressor(max_depth=3)
tree_fit = tree.fit(x_train, y_train)

pred = tree_fit.predict(x_test) # make prediction on test set  
print('RMSE value is:', np.sqrt(mean_squared_error(y_test,pred)))

scores = cross_val_score(tree, X, y, cv=10) # cross-validation scores
print("Performance: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) #estimate mean and variance from cross validation

################################# Visualization of Results (Do not need to know) #################################

# Plot regressor tree
#sort_idx = X.flatten().argsort()
#lin_regplot(X[sort_idx], y[sort_idx], tree)

# Axes labels
#plt.xlabel('Freq.')
#plt.ylabel('Spending in dollars ($)')
#plt.savefig('regression tree.png', dpi=300)
#plt.show() # Display figure

############################### Regressor Tree - Model Optimization - Nested CV #########################################

# Find optimal paramater for DecisionTreeRegressor with GridSearchCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor

# Assign attribute and target variable
X = df[["Freq."]].values # Attribute
y = df[["Spending"]].values # Target Variable

# split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

inner_cv = KFold(n_splits=10, shuffle=True)
outer_cv = KFold(n_splits=10, shuffle=True)

# Find the max_depth that minimizes MSE
# Inner CV
parameters = {'max_depth':range(3,50), 
              'criterion':['mse', 'friedman_mse', 'mae'], 
              'splitter':['best', 'random'], 
              'min_samples_split':range(1,10), 
              'min_samples_leaf':range(1,10)}

gs_dt = GridSearchCV(DecisionTreeRegressor(random_state=42), parameters, n_jobs=4, cv=3, iid = True) #GridSearchCV
gs_dt.fit(x_train, y_train) # Fit model

gs_dt = gs_dt.fit(x_train, y_train)

pred3 = gs_dt.predict(x_test) # make prediction on test set  
print('RMSE value is:', np.sqrt(mean_squared_error(y_test,pred3)))

print("Parameter Tuning")
print("Non-nested Performance: ", gs_dt.best_score_)
print("Optimal Parameter: ", gs_dt.best_params_)    # Parameter setting that gave the best results on the hold out data.
print("Optimal Estimator: ", gs_dt.best_estimator_) # Estimator that was chosen by the search, i.e. estimator which gave highest score

# Outer CV
nested_score_gs_dt = cross_val_score(gs_dt, X=X, y=y, cv=outer_cv)
print("Nested CV Performance: ",nested_score_gs_dt.mean(), " +/- ", nested_score_gs_dt.std())

# # Modeling nonlinear relationships with different models

######################### Increasing Linear Regression Model Complexity ##############################

from sklearn.preprocessing import PolynomialFeatures

X = df[["Freq."]].values # Attribute
y = df[["Spending"]].values # Target Variable

regr = LinearRegression() #Linear Regression instance
knn = neighbors.KNeighborsRegressor(n_neighbors = 3) #kNN instance
rtree = DecisionTreeRegressor(max_depth=3) #Regresion Tree instance

# Generate polynomial and interaction features.

# PolynomialFeatures = Generate a new feature matrix consisting of all polynomial 
# combinations of the features with degree less than or equal to the specified degree. 
# For example, if an input sample is two dimensional and of the form [a, b], the degree-2 
# polynomial features are [1, a, b, a^2, ab, b^2].
quadratic = PolynomialFeatures(degree=2) # degree = the degree of the polynomial features (default = 2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# Data size after polynomial transformation
print(X.shape) 
print(X_quad.shape)
print(X_cubic.shape)

# Preview data after polynomial transformations
print(X[1:5,:]) 
print(X_quad[1:5,:])
print(X_cubic[1:5,:])

##### Regular variables

X_train, X_test, y_train, y_test = train_test_split( # Split Validation
    X, y, test_size=0.3, random_state=42)

# fit features for three different models
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

regr = regr.fit(X_train, y_train) #Fit Linear Model
y_lin_fit = regr.predict(X_fit)   #Apply Model
linear_mse = mean_squared_error(y_test, regr.predict(X_test)) # mean_squared_error test set
print("MSE (linear regression): %0.2f" % linear_mse)

knn = knn.fit(X_train, y_train) #Fit kNN Model
y_knn_fit = knn.predict(X_fit)   #Apply Model
knn_mse = mean_squared_error(y_test, knn.predict(X_test)) # mean_squared_error test set
print("MSE (kNN): %0.2f" % knn_mse)

rtree = rtree.fit(X_train, y_train) #Fit Regression Tree Model
y_rtree_fit = tree.predict(X_fit)   #Apply Model
rtree_mse = mean_squared_error(y_test, rtree.predict(X_test)) # mean_squared_error test set
print("MSE (regression tree): %0.2f" % rtree_mse)

##### Quadratic Transformation

X_quad_train, X_quad_test, y_train, y_test = train_test_split( # Split Validation
    X_quad, y, test_size=0.3, random_state=42)


regr = regr.fit(X_quad_train, y_train) # Fit on Linear Regression
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit)) #Fit Quadratic Model
quadratic_mse = mean_squared_error(y_test, regr.predict(X_quad_test)) # mean_squared_error test set
print("MSE (linear regression - quadratic): %0.2f" % quadratic_mse)

knn = knn.fit(X_quad_train, y_train) # Fit on kNN
y_quad_fit = knn.predict(quadratic.fit_transform(X_fit)) #Fit Quadratic Model 
quadratic_mse = mean_squared_error(y_test, knn.predict(X_quad_test)) # mean_squared_error test set
print("MSE (kNN - quadratic): %0.2f" % quadratic_mse)

rtree = rtree.fit(X_quad_train, y_train) # Fit on Regression Tree
y_quad_fit = rtree.predict(quadratic.fit_transform(X_fit)) #Fit Quadratic Model
quadratic_mse = mean_squared_error(y_test, rtree.predict(X_quad_test)) # mean_squared_error test set
print("MSE (regression tree - quadratic): %0.2f" % quadratic_mse)

##### Cubic Transformation

X_cubic_train, X_cubic_test, y_train, y_test = train_test_split( # Split Validation
    X_cubic, y, test_size=0.3, random_state=42)

regr = regr.fit(X_cubic_train, y_train) # Fit on Linear Regression
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit)) #Fit Cubic Model
cubic_mse = mean_squared_error(y_test, regr.predict(X_cubic_test)) # mean_squared_error test set
print("MSE (linear regression - cubic): %0.2f" % quadratic_mse)

knn = knn.fit(X_cubic_train, y_train) # Fit on kNN
y_cubic_fit = knn.predict(cubic.fit_transform(X_fit)) #Fit Cubic Model
cubic_mse = mean_squared_error(y_test, knn.predict(X_cubic_test)) # mean_squared_error test set
print("MSE (kNN - cubic): %0.2f" % quadratic_mse)

rtree = rtree.fit(X_cubic_train, y_train) # Fit on Regression Tree
y_cubic_fit = rtree.predict(cubic.fit_transform(X_fit)) #Fit Cubic Model
cubic_mse = mean_squared_error(y_test, rtree.predict(X_cubic_test)) # mean_squared_error test set
print("MSE (regression tree - cubic): %0.2f" % quadratic_mse)

######################################## Feature Transformation ###########################################

X = df[["Freq."]].values # Attribute
y = df[["Spending"]].values # Target Variable

# transform features
# The log transformation can be used to make highly skewed distributions less skewed.
X_log = np.log(X)
y_log = np.log(y)

# Sqrt is a transformation with a moderate effect on distribution shape: it is weaker 
# than the logarithm and the cube root. It is also used for reducing right skewness, 
# and also has the advantage that it can be applied to zero values. 
X_sqrt = np.sqrt(X)
y_sqrt = np.sqrt(y)

#### Square Root Transformation

# fit features
X_fit_sqrt = np.arange(X_sqrt.min()-1, X_sqrt.max()+1, 1)[:, np.newaxis]

regr = regr.fit(X_sqrt, y_sqrt) # Fit linear regression model to data with transformed features
y_lin_fit = regr.predict(X_fit_sqrt) # Apply model
linear_mse = mean_squared_error(y_sqrt, regr.predict(X_sqrt)) #MSE
print("MSE (linear regression - sqrt): %0.2f" % linear_mse)

knn = knn.fit(X_sqrt, y_sqrt) # Fit kNN model to data with transformed features
y_lin_fit = knn.predict(X_fit_sqrt) # Apply model
knn_mse = mean_squared_error(y_sqrt, knn.predict(X_sqrt)) #MSE
print("MSE (kNN - sqrt): %0.2f" % knn_mse)

rtree = rtree.fit(X_sqrt, y_sqrt) # Fit regression tree model to data with transformed features
y_lin_fit = rtree.predict(X_fit_sqrt) # Apply model
rtree_mse = mean_squared_error(y_sqrt, rtree.predict(X_sqrt)) #MSE
print("MSE (regression tree - sqrt): %0.2f" % rtree_mse)

################################# Visualization of Results (Optional Step) ############################################

# plot results
plt.scatter(X_sqrt, y_sqrt, label='training points', color='black') # Visualize raw data (x,y)

plt.plot(X_fit_sqrt, y_lin_fit, # Visualize linear model
         label='linear (d=1), $MSE=%.2f$' % linear_mse, 
         color='blue', 
         lw=2)

# Axes labels and legend position
plt.xlabel('sqrt(Freq.)')
plt.ylabel('$\sqrt{Spending \; in \;')
plt.legend(loc='lower left')

plt.tight_layout()
#plt.savefig('linear_nonlinear2.png', dpi=300)
plt.show()