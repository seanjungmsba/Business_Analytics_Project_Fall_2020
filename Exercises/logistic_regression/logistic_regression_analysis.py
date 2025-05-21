# This block imports several packages that are necessary for the model building 
# and simply loads the data from sklearn dataset library.
from __future__ import division, print_function, unicode_literals

# This used a command from the package matplotlib inline to specify that all the graphs should be plotted inline.
%matplotlib inline

# Import other packages that will be required to do Knn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

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



# Calculate summary statistics
df.describe()

# Group by Average
df.groupby("Diagnosis")["radius (mean)", "texture (mean)", "perimeter (mean)", "area (mean)", "smoothness (mean)", "compactness (mean)", "concavity (mean)", "concave points (mean)", "symmetry (mean)", "fractal dimension (mean)"].mean()

# Group by Standard Deviation
df.groupby("Diagnosis")["radius (std)", "texture (std)", "perimeter (std)", "area (std)", "smoothness (std)", "compactness (std)", "concavity (std)", "concave points (std)", "symmetry (std)", "fractal dimension (std)"].mean()

# Group By the Average of Three Largest Sample  
df.groupby("Diagnosis")["radius (largest)", "texture (largest)", "perimeter (largest)", "area (largest)", "smoothness (largest)", "compactness (largest)", "concavity (largest)", "concave points (largest)", "symmetry (largest)", "fractal dimension (largest)"].mean()

# Correlation Matrix
cor = df.corr()

# Get a Visual Representation of the Correlation Matrix using Seaborn and Matplotlib
# Hitmap
import seaborn as sn
import matplotlib.pyplot as plt

sn.heatmap(cor, annot=False)
plt.show()

# Correlation matrix
# https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
plt.matshow(df.corr())

f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)

plt.show()

# Retrieve features/attributes of dataset
X = df.iloc[:,2:31]
X

y = df.iloc[:,1]
y





################################# Function to Visualize Decision Regions of kNN #################################
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# This function visualizes the "decision surfaces" of the kNN algorithm
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

############################################## Split the Data ##############################################
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

########################################## Distribution Target Variable ##########################################
print('Labels counts in y:', pd.value_counts(y))
print('Labels counts in y_train:', pd.value_counts(y_train))
print('Labels counts in y_test:', pd.value_counts(y_test))



# ## Standardizing the features:

############################################# Normalization #############################################
from sklearn.preprocessing import StandardScaler # Standardize features by removing the mean and scaling to unit variance

sc = StandardScaler()
sc.fit(X_train) # Compute the mean and std to be used for later scaling.

X_train_std = sc.transform(X_train) # Perform standardization of train set X by centering and scaling
X_test_std = sc.transform(X_test) # Perform standardization of test set X by centering and scaling

############################################# Train the Model #############################################
from sklearn import neighbors, datasets

# KNeighborsClassifier is a classifier implementing the k-nearest neighbors vote.
# Learn more about it here https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# Set parameters of KNeighborsClassifier
knn = neighbors.KNeighborsClassifier(n_neighbors=3, #n_neighbors is the k in the kNN
                           p=2, 
                           metric='minkowski') #The default metric is minkowski, which is a generalization of the Euclidean distance
                                               # with p=2 is equivalent to the standard Euclidean distance.
                                               # with p=1 is equivalent to the Mahattan distance.

# Train the model      
knn = knn.fit(X_train_std, y_train)  

X_train_std.shape

y_train.shape

####################################### Visualize decision regions #######################################
# Stacks needed for visualization of decision regions
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Visualization of decision regions
plot_decision_regions(X_combined_std, y_combined, 
                      classifier=knn, test_idx=range(398, 569))

# Set parameters for visualization
plt.xlabel('Attibutes [standardized]')
plt.ylabel('Target [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# ValueError: query data dimension must match training data dimension

############################################# Evaluate the Model #############################################
################################################### KNN ######################################################

# The sklearn.metrics module includes score functions, performance metrics and pairwise metrics 
# and distance computations.
# https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
from sklearn.metrics import confusion_matrix

# Estimate the predicted values by applying the kNN algorithm
y_pred = knn.predict(X_test_std)
y_pred_insample = knn.predict(X_train_std)

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Accuracy
print('Accuracy (out-of-sample): %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy (in-sample): %.2f' % accuracy_score(y_train, y_pred_insample))

# F1 score
print('F1 score (out-of-sample): ', f1_score(y_test, y_pred, average='macro'))
print('F1 score (in-sample)    : ', f1_score(y_train, y_pred_insample, average='macro'))

# Kappa score
print('Kappa score (out-of-sample): ', cohen_kappa_score(y_test, y_pred))
print('Kappa score (in-sample)    : ', cohen_kappa_score(y_train, y_pred_insample))

# Build a text report showing the main classification metrics (out-of-sample performance)
print(classification_report(y_test, y_pred))

############################################# Find Nearest Neighbors #############################################

# Finds the K-neighbors of a point. 
# print('The k nearest neighbors (and the corresponding distances) to user [1, 1] are:', knn.kneighbors([[1., 1.]]))


# ValueError: query data dimension must match training data dimension

# Finds the K-neighbors of all points in the training set.
print('The k nearest neighbors to each user are:', knn.kneighbors(X, return_distance=False)) 

# Computes the (weighted) graph of k-Neighbors for points in X (complete training set)
A = knn.kneighbors_graph(X) # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
A.toarray()

# Result looks odd

########################### Visualize decision regions for different k of kNN ###########################

# from sklearn import neighbors, datasets
# Visualization of the decision boundaries


# from sklearn import neighbors, datasets


# for n_neighbors in [1,5,20,50,100]: # Different k values
    #knn = neighbors.KNeighborsClassifier(n_neighbors, p=2, metric='minkowski') 
    #The default metric is minkowski, which is a generalization of the Euclidean distance
    # with p=2 is equivalent to the standard Euclidean distance.
    #knn = knn.fit(X_train_std, y_train)            # with p=1 is equivalent to the Mahattan distance.

    #X_combined_std = np.vstack((X_train_std, X_test_std))
    #y_combined = np.hstack((y_train, y_test))

    #plot_decision_regions(X_combined_std, y_combined, 
                          classifier=knn, test_idx=range(105, 150))

    #plt.xlabel('petal length [standardized]')
    #plt.ylabel('petal width [standardized]')
    #plt.legend(loc='upper left')
    #plt.tight_layout()
    #plt.title("3-Class classification (k = %i, weights = '%s')"
              #% (n_neighbors, 'distance'))
    #plt.show()
    
# ValueError: query data dimension must match training data dimension


#################################### Visualization of Sigmoid Function ####################################

# Import necessary libraries and modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sigmoid Function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# List of values
z = np.arange(-8, 8, 0.1) # Return evenly spaced values within a given interval using the specified step 
phi_z = sigmoid(z)        # Takes z as input and returns sigmoid of z
print(z[0],phi_z[0])      # See first elements of arrays

# Visualization parameters
plt.plot(z, phi_z)          # Specify what to plot
plt.axvline(0.0, color='k') # Add a vertical line across the axes
plt.ylim(-0.1, 1.1)         # Set the y-limits of the current axes
plt.xlabel('z')             # Set label of x axis         
plt.ylabel('$\phi (z)$')    # Set label of y axis

# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout() # tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area.
plt.show()         # Display the figure

# Import urlopen package to read files directly from the websites
from urllib.request import urlopen

link = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
f = urlopen(link)
data = f.read()

feature_name = ["ID number", "Diagnosis", "radius (mean)", "texture (mean)", "perimeter (mean)", "area (mean)", "smoothness (mean)", "compactness (mean)", "concavity (mean)", "concave points (mean)", "symmetry (mean)", "fractal dimension (mean)", "radius (std)", "texture (std)", "perimeter (std)", "area (std)", "smoothness (std)", "compactness (std)", "concavity (std)", "concave points (std)", "symmetry (std)", "fractal dimension (std)", "radius (largest)", "texture (largest)", "perimeter (largest)", "area (largest)", "smoothness (largest)", "compactness (largest)", "concavity (largest)", "concave points (largest)", "symmetry (largest)", "fractal dimension (largest)"]
feature_name

# Converting bytes to Pandas dataframe
# Code adpated from: https://stackoverflow.com/questions/47379476/how-to-convert-bytes-data-into-a-python-pandas-dataframe
from io import StringIO
s=str(data,'utf-8')
data = StringIO(s) 
df=pd.read_csv(data, header=None)
df.columns = feature_name

# Change the column types
df[["radius (mean)", "texture (mean)", "perimeter (mean)", "area (mean)", "smoothness (mean)", "compactness (mean)", "concavity (mean)", "concave points (mean)", "symmetry (mean)", "fractal dimension (mean)", "radius (std)", "texture (std)", "perimeter (std)", "area (std)", "smoothness (std)", "compactness (std)", "concavity (std)", "concave points (std)", "symmetry (std)", "fractal dimension (std)", "radius (largest)", "texture (largest)", "perimeter (largest)", "area (largest)", "smoothness (largest)", "compactness (largest)", "concavity (largest)", "concave points (largest)", "symmetry (largest)", "fractal dimension (largest)"]] = df[["radius (mean)", "texture (mean)", "perimeter (mean)", "area (mean)", "smoothness (mean)", "compactness (mean)", "concavity (mean)", "concave points (mean)", "symmetry (mean)", "fractal dimension (mean)", "radius (std)", "texture (std)", "perimeter (std)", "area (std)", "smoothness (std)", "compactness (std)", "concavity (std)", "concave points (std)", "symmetry (std)", "fractal dimension (std)", "radius (largest)", "texture (largest)", "perimeter (largest)", "area (largest)", "smoothness (largest)", "compactness (largest)", "concavity (largest)", "concave points (largest)", "symmetry (largest)", "fractal dimension (largest)"]].apply(pd.to_numeric)
df['Diagnosis'] = pd.Categorical(df.Diagnosis)

# Calculate summary statistics
df.describe()

# Retrieve features/attributes of dataset
X = df.iloc[:,2:31]
X

y = df.iloc[:,1]
y

# To write a Python 2/3 compatible codebase, the first step is to add this line to the top of each module
from __future__ import division, print_function, unicode_literals

# Import necessary libraries and modules 
# Matplotlib inline allows the output of plotting commands will be displayed inline
%matplotlib inline                      
from sklearn import linear_model # The sklearn.linear_model module implements generalized linear models. LR is part of this module

######################################### Load Libraries and Modules #########################################

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

############################################    Split the Data   ############################################

# Split validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1, stratify=y)

#################################### Train the Logistic Regression Model ####################################

# We create an instance of the Classifier
# Logistic Regression (aka logit) classifier.
clf = linear_model.LogisticRegression(C=1e5) # C parameter is the inverse of regularization strength
                                             # C must be a positive float
                                             # C in this case is 1/lambda
                                             # Smaller values specify stronger regularization
                                             # Applies regularization by default; you can set C very large to avoid regularization (setting penalty l2 can speed up the estimations with a very large C)

# Train the model (fit the data)
# As with other classifiers, DecisionTreeClassifier takes as input two arrays: an array X, sparse or dense, 
# of size [n_samples, n_features] holding the training samples, and an array Y of integer values, size [n_samples], 
# holding the class labels for the training samples:
clf = clf.fit(X_train, y_train)
print('The weights of the attributes are:', clf.coef_)

#################################### Apply the Logistic Regression Model ####################################

y_pred = clf.predict(X_test)             # Classification prediction
y_pred_prob = clf.predict_proba(X_test)  # Class probabilities
print(y_pred[0], y_pred_prob[0], np.sum(y_pred_prob[0]))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

################################### Evaluate the Logistic Regression Model ##################################

# Build a text report showing the main classification metrics (out-of-sample performance)
print(classification_report(y_test, y_pred))



# # Applying the logistic regression model

#################################### Apply the Logistic Regression Model ####################################

# After being fitted, the model can then be used to predict the class of samples:
print('The 1st instance is predicted to belong to class:', clf.predict(df.iloc[:1, 2:31]))

# Alternatively, the probability of each class can be predicted, which is the fraction of training samples of the same class in a leaf:
print('The probabilities of belonging to each one of the classes are estimated as:', clf.predict_proba(df.iloc[:1, 2:31]))

# We can also try clf.decision_function(X)
# The desion function tells us on which side of the hyperplane generated by the classifier we are 
# (and how far we are away from it). Based on that information, the estimator then label the examples 
# with the corresponding label.



# # Visualizing the logistic regression boundaries

# Function that will help us visualize the decision surfaces
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2= np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

from sklearn.linear_model import LogisticRegression
#from mlxtend.plotting import plot_decision_regions

# Function that will help us visualize the decision surfaces
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

lr = LogisticRegression(C=1e5, random_state=1)
lr.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined,
                      classifier=lr, test_idx=range(398, 569))

plt.xlabel('Attibutes [standardized]')
plt.ylabel('Target [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# ValueError: X has 2 features per sample; expecting 29



# # Estimating Generalization Performance with Cross-Validation

#################################### Logistic Regression with Cross Validation ####################################

from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Fit model to all the data
clf_lr = linear_model.LogisticRegression(C=1)

# Evaluate performance with cross-validation
# Read more about cross_val_score in the following link 
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score

# Accuracy
scores=cross_val_score(clf_lr, df.iloc[:, 2:31], df.iloc[:, 1], cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

# F-1 scores
scores_f1=cross_val_score(clf_lr, df.iloc[:, 2:31], df.iloc[:, 1], cv=10, scoring='f1_macro')
print("F1-score: %0.2f (+/- %0.2f)" % (scores_f1.mean(), scores_f1.std() * 2))# returns an array of scores of the estimator for each run of the cross validation.
print(scores_f1)

# Use all features of the data
scores = cross_val_score(clf_lr, df.iloc[:, 2:31], df.iloc[:, 1], cv=10, scoring='f1_macro')
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



# # Learning Curves

################################## Define function that plots Learning Curves ##################################

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)): # np.linspace(.1, 1.0, 5) will return evenly
                                                                        # spaced 5 numbers from 0.1 to 1.0
                        # n_jobs is the number of CPUs to use to do the computation. 
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
    
    # Visualization patamters
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    # Estimate train and test score for different training set sizes
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes) # learning_curve Determines cross-validated 
                                                                        # training and test scores for different 
                                                                        # training set sizes.

    # Estimate statistics of train and test scores (mean, std)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    # Fill the area around the mean scores with standard deviation info
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r") # Fill for train set scores

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")  # Fill for test set scores
    
    # Visualization parameters that will allow us to distinguish train set scores from test set scores
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

###################################### Plot Learning Curves (LR and kNN) #######################################

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors

title = "Learning Curve (LR)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
estimator = LogisticRegression()
plot_learning_curve(estimator, title, X, y, (0.0, 1.01), cv=cv, n_jobs=4)

plt.show()

title = "Learning Curve (kNN)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
estimator = neighbors.KNeighborsClassifier() #n_neighbors=
plot_learning_curve(estimator, title, X, y, (0.0, 1.01), cv=cv, n_jobs=4)

plt.show()