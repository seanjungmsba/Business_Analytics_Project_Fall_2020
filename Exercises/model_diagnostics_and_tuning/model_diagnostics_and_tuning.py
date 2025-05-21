############################## Python version compatibility ##############################

# To write a Python 2/3 compatible codebase, the first step is to add this line to the top of each module
from __future__ import division, print_function, unicode_literals

############################## Import Libraries & Modules ################################

# This used a command from the package matplotlib inline to specify that all the graphs should be plotted inline.
%matplotlib inline

# Import necessary libraries and specify that graphs should be plotted inline. 
from sklearn import linear_model       # The sklearn.linear_model module implements generalized linear models
import numpy as np                     # NumPy is the package for scientific computing with Python
import pandas as pd                    # Pandas is the package for working with data frame

# Also, import urlopen package to read files directly from the websites
from urllib.request import urlopen

link = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
f = urlopen(link)
data = f.read()
print(data)
print(type(data))

feature_name = ["ID number", "Diagnosis", "radius (mean)", "texture (mean)", "perimeter (mean)", "area (mean)", "smoothness (mean)", "compactness (mean)", "concavity (mean)", "concave points (mean)", "symmetry (mean)", "fractal dimension (mean)", "radius (std)", "texture (std)", "perimeter (std)", "area (std)", "smoothness (std)", "compactness (std)", "concavity (std)", "concave points (std)", "symmetry (std)", "fractal dimension (std)", "radius (largest)", "texture (largest)", "perimeter (largest)", "area (largest)", "smoothness (largest)", "compactness (largest)", "concavity (largest)", "concave points (largest)", "symmetry (largest)", "fractal dimension (largest)"]
feature_name

# Converting bytes to Pandas dataframe
# Code adpated from: https://stackoverflow.com/questions/47379476/how-to-convert-bytes-data-into-a-python-pandas-dataframe
from io import StringIO
s=str(data,'utf-8')
data = StringIO(s) 
df=pd.read_csv(data, header=None)
df.columns = feature_name
df.head()

# Change the column types
df[["radius (mean)", "texture (mean)", "perimeter (mean)", "area (mean)", "smoothness (mean)", "compactness (mean)", "concavity (mean)", "concave points (mean)", "symmetry (mean)", "fractal dimension (mean)", "radius (std)", "texture (std)", "perimeter (std)", "area (std)", "smoothness (std)", "compactness (std)", "concavity (std)", "concave points (std)", "symmetry (std)", "fractal dimension (std)", "radius (largest)", "texture (largest)", "perimeter (largest)", "area (largest)", "smoothness (largest)", "compactness (largest)", "concavity (largest)", "concave points (largest)", "symmetry (largest)", "fractal dimension (largest)"]] = df[["radius (mean)", "texture (mean)", "perimeter (mean)", "area (mean)", "smoothness (mean)", "compactness (mean)", "concavity (mean)", "concave points (mean)", "symmetry (mean)", "fractal dimension (mean)", "radius (std)", "texture (std)", "perimeter (std)", "area (std)", "smoothness (std)", "compactness (std)", "concavity (std)", "concave points (std)", "symmetry (std)", "fractal dimension (std)", "radius (largest)", "texture (largest)", "perimeter (largest)", "area (largest)", "smoothness (largest)", "compactness (largest)", "concavity (largest)", "concave points (largest)", "symmetry (largest)", "fractal dimension (largest)"]].apply(pd.to_numeric)
df['Diagnosis'] = pd.Categorical(df.Diagnosis)

# Retrieve features/attributes of dataset
X = df.iloc[:,2:31]
X

y = df.iloc[:,1]
y

df.head()

############################## Function for Learning Curves ##############################

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()                    #display figure
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training sizes") #y label title
    plt.ylabel("Score")             #x label title
    
    # Class learning_curve determines cross-validated training and test scores for different training set sizes
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    # Cross validation statistics for training and testing data (mean and standard deviation)
    train_scores_mean = np.mean(train_scores, axis=1) # Compute the arithmetic mean along the specified axis.
    train_scores_std = np.std(train_scores, axis=1)   # Compute the standard deviation along the specified axis.
    test_scores_mean = np.mean(test_scores, axis=1)   # Compute the arithmetic mean along the specified axis.
    test_scores_std = np.std(test_scores, axis=1)     # Compute the standard deviation along the specified axis.

    plt.grid() # Configure the grid lines

    # Fill the area around the line to indicate the size of standard deviations for the training data
    # and the test data
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r") # train data performance indicated with red
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g") # test data performance indicated with green
    
    # Cross-validation means indicated by dots
    # Train data performance indicated with red
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="In-Sample Performance Score")
    # Test data performance indicated with green
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Out-Sample Performance Score")

    plt.legend(loc="best") # Show legend of the plot at the best location possible
    return plt             # Function that returns the plot as an output

########################### Visualization of Learning Curves ###########################

# Determines cross-validated training and test scores for different training set sizes
from sklearn.model_selection import learning_curve 
# Random permutation cross-validator
from sklearn.model_selection import ShuffleSplit
# Logistic regression classifier class
from sklearn.linear_model import LogisticRegression
# kNN classifier class
from sklearn import neighbors
# Each pyplot function makes some change to a figure: e.g., creates a figure, creates a plotting area in a figure, 
# plots some lines in a plotting area, decorates the plot with labels, etc
import matplotlib.pyplot as plt


title = "Learning Curve (Logistic Regression) by Sean Jung"

# Class ShuffleSplit is a random permutation cross-validator
# Parameter n_splits = Number of re-shuffling & splitting iterations
# Parameter test_size = represents the proportion of the dataset to include in the test split (float between 0.0 and 1.0) 
# Parameter random_state = the seed used by the random number generator
cv = ShuffleSplit(n_splits=20, test_size=0.3, random_state=42)
estimator = LogisticRegression() # Build multiple LRs as we increase the size of the traning data
# Plots the learning curve based on the previously defined function for the logistic regression estimator
plot_learning_curve(estimator, title, X, y, (0.89, 1.01), cv=cv, n_jobs=4)

plt.show() # Display the figure


############################################################################################

#title = "Learning Curve (kNN)"

# Plots the learning curve based on the previously defined function for the kNN classifier. Uses the 
# random permutation cross-validator
#cv = ShuffleSplit(n_splits=20, test_size=0.3, random_state=42)
#estimator = neighbors.KNeighborsClassifier() #n_neighbors=5 by default
#plot_learning_curve(estimator, title, X, y, (0.89, 1.01), cv=cv, n_jobs=4)

#plt.show() # Display the figure



# # Addressing overfitting and underfitting with fitting graphs

############################## Import Libraries & Modules ################################

# Logistic regression classifier class
from sklearn.linear_model import LogisticRegression
# pandas is a library providing high-performance, easy-to-use data structures and data 
# analysis tools for the Python programming language
import pandas as pd 
# Evaluate a score by cross-validation
from sklearn.model_selection import cross_val_score
# Encode labels with value between 0 and n_classes-1
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

###################################### Classifier ######################################

le = LabelEncoder()
y = le.fit_transform(y)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
# Class for Logistic Regression (aka logit) classifier.
# Parameter C: Inverse of regularization strength; 
# C must be a positive float; smaller values specify stronger regularization.
clf_lr = linear_model.LogisticRegression(C=1e4, solver='liblinear') # default: ‘l2’ norm

############################ Performance w/ Cross Validation ############################

# Evaluate performance
# Read more about cross_val_score in the following link 
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score
# Possible 'scoring' values http://scikit-learn.org/stable/modules/model_evaluation.html
scores=cross_val_score(clf_lr, X=X, y=y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores) # print accuracy for each iteration of cross-validation

scores_f1=cross_val_score(clf_lr, X=X, y=y, cv=10, scoring='f1_macro')
print("F1-score: %0.2f (+/- %0.2f)" % (scores_f1.mean(), scores_f1.std() * 2)) # returns an array of scores of the estimator for each run of the cross validation.
print(scores_f1) # print f1-score for each iteration of cross-validation

############################### Import Libraries & Modules #################################

# Fitting curve (aka validation curve)
# Determine training and test scores for varying parameter values.
from sklearn.model_selection import validation_curve
# Split validation
from sklearn.model_selection import train_test_split
# Class for Logistic Regression classifier
from sklearn.linear_model import LogisticRegression 
# Class for Decision Tree classfier
#from sklearn.tree import DecisionTreeClassifier

np.random.seed(42) #the seed used by the random number generator for np

############################# Parameters - Varying Complexity #############################

# Specify possible parameter values for max_depth.
# Parameter max_depth: depth of the decision tree; 
# It must be a positive integers
max_depth = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# Compute scores for an estimator with different values of a specified parameter. 
# This is similar to grid search with one parameter. 
# However, this will also compute training scores and is merely a utility for plotting the results.

########################## Estimate Scores - Varying Complexity ##########################

# Determine training and test scores for varying parameter values.
train_scores, test_scores = validation_curve( 
                estimator=DecisionTreeClassifier(criterion = 'entropy', random_state=42), #Build Decison Tree Models
                X=X_train, 
                y=y_train, 
                param_name='max_depth',
                param_range=max_depth,
                cv=20,  #20-fold cross-validation
                scoring='accuracy',
                n_jobs=4) # Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”. This parameter is ignored when the ``solver``is set to ‘liblinear’ regardless of whether ‘multi_class’ is specified or not. If given a value of -1, all cores are used.


# Cross validation statistics for training and testing data (mean and standard deviation)
train_mean = np.mean(train_scores, axis=1) # Compute the arithmetic mean along the specified axis.
train_std = np.std(train_scores, axis=1)   # Compute the standard deviation along the specified axis.
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

############################## Visualization - Fitting Graph ##############################

# Plot train accuracy means of cross-validation for all the parameters C in param_range
plt.plot(max_depth, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='In-Sample Accuracy')

# Fill the area around the line to indicate the size of standard deviations of performance for the training data
plt.fill_between(max_depth, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

# Plot test accuracy means of cross-validation for all the parameters C in param_range
plt.plot(max_depth, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='Out-Sample Accuracy')

#Fill the area around the line to indicate the size of standard deviations of performance for the test data
plt.fill_between(max_depth, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

# Grid and Axes Titles
plt.grid()
#plt.xscale('linear')
plt.legend(loc='lower right')
plt.xlabel('Number of nodes')
plt.ylabel('Accuracy')
plt.ylim([0.90, 1.01]) # y limits in the plot
plt.tight_layout()
# plt.savefig('Fitting_graph_LR.png', dpi=300)
plt.title('Fitting Graph for Decision Tree by Sean Jung')
plt.show()    # Display the figure


# Similar to the learning_curve function, the validation_curve function (i.e., the function that plots the fitting graphs) uses the stratified k-fold cross-validation by default to estimate the performance of the model if we are using algorithms for the classification. Inside the validation_curve function, we specified the parameter that we wanted to evaluate. In this case, it is  C , the inverse regularization parameter of the LogisticRegression classifier to assess the LogisticRegression object for a specific value range that we set via the param_range parameter. Similar to the learning curve example in the previous section, we plotted the average training and cross-validation accuracies and the corresponding standard deviations.

# Although the differences in the accuracy for varying values of  C  are subtle, we can see that the model slightly underfits the data when we increase the regularization strength (small values of  C ). However, for large values of  C , it means lowering the strength of regularization, so the model tends to slightly overfit the data. In this case, the sweet spot appears to be around  C=1000 .

# # Fine-tuning machine learning models via grid search

# In machine learning, we have two types of paramaters: (1) those learned from the training data, for example, the weights in logistic regression, and (2) the parameters of a learning algorithm that are optimized separately. The latter are the tunning parameters, also called hyperparameters, of a model, for example, the regularization parameter in logistic regression or the depth of a decision tree.
# In the previous section, we used validation curves to improve the performance of a model by tuning one of its hyperparameters. In this section, we will take a look at a powerful hyperparameter optimization technique called grid search that can further help to improve the performance of a model by finding the * *optimal combination of hyperparameter values**.

############################### Import Libraries & Modules #################################
from sklearn.tree import DecisionTreeClassifier # A decision tree classifier
# GridSearchCV performs an exhaustive search over specified parameter values for an estimator
# The parameters of the estimator used to apply these methods are optimized by cross-validated 
# grid-search over a parameter grid.
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score #http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn.linear_model import LogisticRegression 
from sklearn import neighbors, datasets
# Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler

np.random.seed(42) # Ensure reproducability

################################# Nested Cross-Validation #################################

##################################### Parameter Tuning ####################################

# Exhaustive search over specified parameter values for an estimator.
# GridSearchCV implements a “fit” and a “score” method. 
# It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” 
# if they are implemented in the estimator used.

# The parameters of the estimator used to apply these methods are optimized by cross-validated 
# grid-search over a parameter grid.

inner_cv = KFold(n_splits=10, shuffle=True)
outer_cv = KFold(n_splits=10, shuffle=True)

############################## Decision Tree Parameter Tuning ##############################

##############################################################################################################
# Choosing optimal depth of the tree
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                  param_grid=[{'max_depth': [1,2,3,4,5,6,7,8,9,10,15,50,100,None]}],
                  scoring='accuracy', # Specifying multiple metrics for evaluation
                  cv=inner_cv)

gs = gs.fit(X,y)
print(" Parameter Tuning #1")
print("Non-nested CV Accuracy: ", gs.best_score_)
print("Optimal Parameter: ", gs.best_params_)    # Parameter setting that gave the best results on the hold out data.
print("Optimal Estimator: ", gs.best_estimator_) # Estimator that was chosen by the search, i.e. estimator which gave highest score
nested_score_gs = cross_val_score(gs, X=X, y=y, cv=outer_cv)
print("Nested CV Accuracy: ",nested_score_gs.mean(), " +/- ", nested_score_gs.std())
##############################################################################################################



##############################################################################################################
# See all the parameters you can optimize here http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# Choosing optimal depth of the tree AND optimal splitting criterion
gs_dt = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                  param_grid=[{'max_depth': [1,2,3,4,5,6,7,8,9,10,15,50,100,None], 
                               'criterion':['gini','entropy']}],
                  scoring='accuracy',
                  cv=inner_cv)

gs_dt = gs_dt.fit(X,y)
print("\n Parameter Tuning #2")
print("Non-nested CV Accuracy: ", gs_dt.best_score_)
print("Optimal Parameter: ", gs_dt.best_params_)
print("Optimal Estimator: ", gs_dt.best_estimator_)
nested_score_gs_dt = cross_val_score(gs_dt, X=X, y=y, cv=outer_cv)
print("Nested CV Accuracy: ",nested_score_gs_dt.mean(), " +/- ", nested_score_gs_dt.std())
###############################################################################################################


##############################################################################################################
# Choosing depth of the tree AND splitting criterion AND min_samples_leaf AND min_samples_split ##############
gs_dt2 = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                  param_grid=[{'max_depth': [1,2,3,4,5,6,7,8,9,10,15,50,100,None], 
                               'criterion':['gini','entropy'], 
                              'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,20,None],
                              'min_samples_split':[2,3,4,5,6,7,8,9,10,20,None]}],
                  scoring='accuracy',
                  cv=inner_cv,
                  n_jobs=4)

gs_dt2 = gs_dt2.fit(X,y)
print("\n Parameter Tuning #3")
print("Non-nested CV Accuracy: ", gs_dt2.best_score_)
print("Optimal Parameter: ", gs_dt2.best_params_)
print("Optimal Estimator: ", gs_dt2.best_estimator_)
nested_score_gs_dt2 = cross_val_score(gs_dt2, X=X, y=y, cv=outer_cv)
print("Nested CV Accuracy: ",nested_score_gs_dt2.mean(), " +/- ", nested_score_gs_dt2.std())
##################################################################################################################


##############################################################################################################
# Choosing depth of the tree AND splitting criterion AND min_samples_leaf AND min_samples_split AND max_features
gs_dt3 = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                  param_grid=[{'max_depth': [1,2,3,4,5,6,7,8,9,10,None], 
                               'criterion':['gini','entropy'], 
                              'min_samples_leaf':[1,2,3,4,5,None],
                              'min_samples_split':[1,2,3,4,5,None],
                              'max_features':[1, 2, 3, 4, 5,None]}],
                  scoring='accuracy',
                  cv=inner_cv,
                  n_jobs=4)

gs_dt3 = gs_dt3.fit(X,y)
print("\n Parameter Tuning #4")
print("Non-nested CV Accuracy: ", gs_dt3.best_score_)
print("Optimal Parameter: ", gs_dt3.best_params_)
print("Optimal Estimator: ", gs_dt3.best_estimator_)
nested_score_gs_dt3 = cross_val_score(gs_dt3, X=X, y=y, cv=outer_cv)
print("Nested CV Accuracy: ",nested_score_gs_dt3.mean(), " +/- ", nested_score_gs_dt3.std())
##############################################################################################################

##############################################################################################################
# Choosing depth of the tree AND splitting criterion AND criterion AND min_samples_leaf AND min_weight_fraction_leaf 
# AND max_leaf_nodesint AND min_samples_split AND min_impurity_decrease AND max_features
gs_dt4 = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
                               'splitter':['best', 'random'],
                               'criterion':['gini','entropy'], 
                              'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,20,30,None],
                               'min_weight_fraction_leaf':[1,2,3,4,5,6,7,8,9,10,None],
                               'max_leaf_nodes':[1,2,3,4,5,6,7,8,9,10,None],
                              'min_samples_split':[0,1,2,3,4,5,6,7,8,9,10,None],
                               'min_impurity_decrease':[0,1,2,3,4,5,6,7,8,9,10,None],
                              'max_features':[1, 2, 3, "auto", "sqrt", "log2"]}],
                  scoring='accuracy',
                  cv=inner_cv,
                  n_jobs=4)

gs_dt4 = gs_dt4.fit(X,y)
print("\n Parameter Tuning #5")
print("Non-nested CV Accuracy: ", gs_dt4.best_score_)
print("Optimal Parameter: ", gs_dt4.best_params_)
print("Optimal Estimator: ", gs_dt4.best_estimator_)
nested_score_gs_dt4 = cross_val_score(gs_dt4, X=X, y=y, cv=outer_cv)
print("Nested CV Accuracy: ",nested_score_gs_dt4.mean(), " +/- ", nested_score_gs_dt4.std())
##############################################################################################################



############################ Logistic Regression Parameter Tuning ############################
#To ignore the convergence warnings
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# Choosing C parameter (i.e., regularization parameter) for Logistic Regression
gs_lr = GridSearchCV(estimator=LogisticRegression(random_state=42),
                  param_grid=[{'C': [0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.1 ,1 ,10, 50, 
                                     100, 500, 1000,2500, 5000, 7500,10000, 15000, 20000, 100000],
                             'solver':['liblinear', 'sag', 'saga']}],
                  scoring='accuracy',
                  cv=inner_cv)




gs_lr = gs_lr.fit(X,y)
print("\n Parameter Tuning #6")
print("Non-nested CV Accuracy: ", gs_lr.best_score_)
print("Optimal Parameter: ", gs_lr.best_params_)
print("Optimal Estimator: ", gs_lr.best_estimator_)
nested_score_gs_lr = cross_val_score(gs_lr, X=X, y=y, cv=outer_cv)
print("Nested CV Accuracy:",nested_score_gs_lr.mean(), " +/- ", nested_score_gs_lr.std())

##############################################################################################################
      
# Choosing C parameter for Logistic Regression AND type of penalty (ie., l1 vs l2)
# See other parameters here http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
gs_lr2 = GridSearchCV(estimator=LogisticRegression(random_state=42, solver='liblinear'),
                  param_grid=[{'C': [5000, 10000, 15000]}],
                  scoring='accuracy',
                  cv=inner_cv)


gs_lr2 = gs_lr2.fit(X,y)
print("\n Parameter Tuning #7")
print("Non-nested CV Accuracy: ", gs_lr2.best_score_)
print("Optimal Parameter: ", gs_lr2.best_params_)
print("Optimal Estimator: ", gs_lr2.best_estimator_)
nested_score_gs_lr2 = cross_val_score(gs_lr2, X=X, y=y, cv=outer_cv)
print("Nested CV Accuracy:",nested_score_gs_lr2.mean(), " +/- ", nested_score_gs_lr2.std())

##############################################################################################################

################################### kNN Parameter Tuning ###################################

#Normalize Data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Choosing k for kNN
gs_knn = GridSearchCV(estimator=neighbors.KNeighborsClassifier(p=2, 
                           metric='minkowski'),
                  param_grid=[{'n_neighbors': [1,3,5,7,9,11,13,15,17,19,21]}],
                  scoring='accuracy',
                  cv=inner_cv)
#print(len(y))
gs_knn = gs_knn.fit(X,y) 
print("\n Parameter Tuning #8")
print("Non-nested CV Accuracy: ", gs_knn.best_score_)
print("Optimal Parameter: ", gs_knn.best_params_)
print("Optimal Estimator: ", gs_knn.best_estimator_)
nested_score_gs_knn = cross_val_score(gs_knn, X=X, y=y, cv=outer_cv)
print("Nested CV Accuracy: ",nested_score_gs_knn.mean(), " +/- ", nested_score_gs_knn.std())
 

#############################################################################################
    
# Choosing k for kNN AND type of distance
gs_knn2 = GridSearchCV(estimator=neighbors.KNeighborsClassifier(p=2, 
                           metric='minkowski'),
                  param_grid=[{'n_neighbors': [1,3,5,7,9,11,13,15,17,19,21],
                               'weights':['uniform','distance']}],
                  scoring='accuracy',
                  cv=inner_cv,
                  n_jobs=4)

gs_knn2 = gs_knn2.fit(X,y)  
print("\n Parameter Tuning #9")
print("Non-nested CV Accuracy: ", gs_knn2.best_score_)
print("Optimal Parameter: ", gs_knn2.best_params_)
print("Optimal Estimator: ", gs_knn2.best_estimator_) # Estimator that was chosen by the search, i.e. estimator which gave highest score
nested_score_gs_knn2 = cross_val_score(gs_knn2, X=X, y=y, cv=outer_cv)
print("Nested CV Accuracy: ",nested_score_gs_knn2.mean(), " +/- ", nested_score_gs_knn2.std())

#############################################################################################


#############################################################################################
    
# Choosing k for kNN AND type of distance AND Algorithm AND leaf size
gs_knn3 = GridSearchCV(estimator=neighbors.KNeighborsClassifier(p=2, 
                           metric='minkowski'),
                  param_grid=[{'n_neighbors': [1,3,5,7,9,11,13,15,17,19,21],
                               'weights':['uniform','distance'],
                              'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                             'leaf_size':[1,5,10,15,30,40,50]}],
                  scoring='accuracy',
                  cv=inner_cv,
                  n_jobs=4)

gs_knn3 = gs_knn3.fit(X,y)  
print("\n Parameter Tuning #10")
print("Non-nested CV Accuracy: ", gs_knn3.best_score_)
print("Optimal Parameter: ", gs_knn3.best_params_)
print("Optimal Estimator: ", gs_knn3.best_estimator_) # Estimator that was chosen by the search, i.e. estimator which gave highest score
nested_score_gs_knn3 = cross_val_score(gs_knn3, X=X, y=y, cv=outer_cv)
print("Nested CV Accuracy: ",nested_score_gs_knn3.mean(), " +/- ", nested_score_gs_knn3.std())

#############################################################################################


#############################################################################################
    
# Choosing k for kNN AND type of distance AND Algorithm AND p
gs_knn4 = GridSearchCV(estimator=neighbors.KNeighborsClassifier(metric='minkowski'),
                  param_grid=[{'n_neighbors': [1,3,5,7,9,11,13,15,17,19,21],
                               'weights':['uniform','distance'],
                              'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                               'p':[1,2]}],
                  scoring='accuracy',
                  cv=inner_cv,
                  n_jobs=4)

gs_knn4 = gs_knn4.fit(X,y)  
print("\n Parameter Tuning #11")
print("Non-nested CV Accuracy: ", gs_knn4.best_score_)
print("Optimal Parameter: ", gs_knn4.best_params_)
print("Optimal Estimator: ", gs_knn4.best_estimator_) # Estimator that was chosen by the search, i.e. estimator which gave highest score
nested_score_gs_knn4 = cross_val_score(gs_knn4, X=X, y=y, cv=outer_cv)
print("Nested CV Accuracy: ",nested_score_gs_knn4.mean(), " +/- ", nested_score_gs_knn4.std())

#############################################################################################


# # Plotting a ROC graph

############################### Import Libraries & Modules #################################

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

######################################## Classifiers ########################################
# Logistic Regression Classifier
clf1 = LogisticRegression(C=2500, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=42, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)

# Decision Tree Classifier
clf2 = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=3, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=42, splitter='best')

# kNN Classifier
clf3 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=13, p=2,
                     weights='distance')


# Label the classifiers
clf_labels = ['Logistic regression', 'Decision tree', 'kNN']
all_clf = [clf1, clf2, clf3]

#################################### Cross - Validation ####################################


print('10-fold cross validation:\n')
# Note: We are assuming here that the data is standardized. For the homework, you need to make sure the data is standardized.
for clf, label in zip([clf1, clf2, clf3], clf_labels): #For all classifiers 
    scores = cross_val_score(estimator=clf,  #Estimate AUC based on cross validation
                             X=X,
                             y=y,
                             cv=10,
                             scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]" #Print peformance statistics based on cross-validation
          % (scores.mean(), scores.std(), label))



##################################### Visualization ######################################

colors = [ 'orange', 'blue', 'green']      # Colors for visualization
linestyles = [':', '--', '-.', '-']        # Line styles for visualization
for clf, label, clr, ls in zip(all_clf,
               clf_labels, colors, linestyles):

    # Assuming the label of the positive class is 1 and data is normalized
    y_pred = clf.fit(X_train,
                     y_train).predict_proba(X_test)[:, 1] # Make predictions based on the classifiers
    fpr, tpr, thresholds = roc_curve(y_true=y_test, # Build ROC curve
                                     y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)                # Compute Area Under the Curve (AUC) 
    plt.plot(fpr, tpr,                         # Plot ROC Curve and create label with AUC values
             color=clr,
             linestyle=ls,
             label='%s (auc = %0.2f)' % (label, roc_auc))

plt.legend(loc='lower right')    # Where to place the legend
plt.plot([0, 1], [0, 1], # Visualize random classifier
         linestyle='--',
         color='gray',
         linewidth=2)

plt.xlim([-0.1, 1.1])   #limits for x axis
plt.ylim([-0.1, 1.1])   #limits for y axis
plt.grid(alpha=0.5)
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.title('ROC Curves by Sean Jung')


#plt.savefig('ROC_all_classifiers', dpi=300)
plt.show()