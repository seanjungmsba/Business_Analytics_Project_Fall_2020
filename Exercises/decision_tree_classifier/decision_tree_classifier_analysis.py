# # 🌳 Decision Tree Classifier: Data Exploration & Modeling
#
# This notebook walks through the process of exploring a dataset and building a **Decision Tree** classifier using Python. We’ll cover the following steps:
#
# 1. 📊 Data Loading & Inspection  
# 2. 🧹 Data Cleaning & Preprocessing  
# 3. 🔍 Exploratory Data Analysis  
# 4. 🌲 Building a Decision Tree Model  
# 5. 🧪 Model Evaluation  
# 6. 📈 Visualizing Results  
# 7. ✅ Conclusions & Next Steps  

# # Data Exploration

# Decision Trees (DTs) are a supervised learning method 
# that can be used for classification. 

# The goal is to create a model that predicts the value of a target variable 
# by learning simple decision rules inferred from the data features

# Setup
# First, let's make sure this notebook works well in both python 2 and 3,
# import a few common modules, ensure MatplotLib plots figures inline and 
# prepare a function to save the figures:

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"

# Define path of an image
# Function that takes as input fig_id and returns back the image path
def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

# Save image in a specific path
# Function that takes as input fig_id and saves the image in a specific format (i.e., png, 300 dpi)
def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)


# Training and Visualizing a Decision Tree

########################################### Imports ###########################################
from sklearn.tree import DecisionTreeClassifier # The sklearn.tree module includes decision tree-based models for 
# classification and regression
# Documentation for decision Tree Classifier 
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# Seaborn is a Python data visualization library based on matplotlib. 
# Seaborn documentation can be found here https://seaborn.pydata.org/generated/seaborn.set.html
import seaborn as sns # sns is an alias pointing to seaborn
sns.set(color_codes=True) #Set aesthetic parameters in one step. Remaps the shorthand color codes (e.g. “b”, “g”, “r”, etc.) to the colors from this palette.
from scipy import stats #Documentation stats package of scipy https://docs.scipy.org/doc/scipy/reference/stats.html#module-scipy.stats

######################################### Data Exploration #########################################
#Load data_set
df = pd.read_csv(r'C:\Users\seanj\Desktop\Intro to Business Analytics\HW1\HW1_Data.csv')
df.head() #df.head() is used to check the first few rows of the dataset

# Explore the data
df.describe()

# Drop the NAs on the target variable, churndep
df.dropna(subset=['churndep'])

# We notice that there are some negative values on columns, 'eqpdays' and 'revenue'
# I am going to filter out negative values of 'eqpdays' and 'revenue'
df = df[df['eqpdays'] >= 0] 
df = df[df['revenue'] >= 0]

# Explore the data again
df.describe()

df.hist(column='churndep')
# The distribution of the two classes is almost equal.

# keep specific attributs (Include all the x variables except the target variable in the last column)
X = df.iloc[:,0:-1] 
# or X = df.iloc[:,:11]
feature= ["revenue", "outcalls","incalls", "months", "eqpdays", "webcap", "marryyes", "travel", "pcown", "creditcd", "retcalls"]

# Retriving Target Variable
y = df.iloc[:,11]

# Exploring Target Variable
# I want to find out the number of each instances in the target variable
unique, counts = np.unique(y, return_counts=True) # 3 distinct classes equally represented
print("The frequency of instances per class in target variable is: " , dict(zip(unique, counts)))
print("The names of the two distinct classes in target variables are: ", list(y.unique())) 

# We can also check if a feature has a very skewed distribution of classes. 
# For example, to plot a histogram of revenue, we can do the following:
df.groupby('churndep').revenue.hist(alpha=0.7)
# To plot the other attributes, replace ‘revenue’ with the other attribute within our model.

import seaborn as sns
corrmatrix = df.corr()
top_correlated_features = corrmatrix.index
plt.figure(figsize=(12,12))

# plot heat map
plot = sns.heatmap(df[top_correlated_features].corr(), annot=True, cmap="RdYlGn")

# A pairplot plot a pairwise relationships in a dataset. 
sns.set()
sns.pairplot(df)

# # Model Training

# The following code trains a DecisionTreeClassifier on the HW1 dataset
# Decision Tree Induction (Fitting the Model)
# Documentation https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# split validation
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.30, random_state=1)

# After properly splitting the data we can finally start modeling and testing out which parameters should be tuned to improve model performance. 
# In order to compare different models we will look at Accuracy of the test set as a benchmark.

# try different max_leaf_nodes parameters:
for i in range(2,30,1):
    clf1 = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, max_leaf_nodes=i)
    clf1 = clf1.fit(X_train, y_train)
    y_pred1 = clf1.predict(X_test)
    rate = accuracy_score(y_test, y_pred1)
    print(str(i) + " " + str(rate))
    
    # Pick the one with highest number (accuracy)

# try different min_impurity_decrease values:
import numpy as np

for i in np.arange(0,0.011,0.001):
    clf2 = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_impurity_decrease=i)
    clf2 = clf2.fit(X_train, y_train)
    y_pred2 = clf1.predict(X_test)
    rate = accuracy_score(y_test, y_pred2)
    print(str(i) + " " + str(rate))
    
    # Pick the one with highest number (accuracy)



######################################### Imports #########################################
from sklearn.tree import export_graphviz
# If you don't have graphviz package, you need to install it https://anaconda.org/anaconda/graphviz
# How to install Graphviz with Anaconda https://anaconda.org/anaconda/graphviz
# conda install -c anaconda graphviz 

from IPython.display import Image

# ATTENTION: You need to change the working directory
# For instance, I had to change it to:
#os.chdir("/Users/vtodri/Dropbox/Vilma/Teaching/Emory/Fall/MSBA/Lectures/Week 2")
os.chdir("/Users/seanj/Desktop/Intro to Business Analytics/HW1")

# make the images and decision_trees path in order for image_path to work
if not os.path.exists('./images/decision_trees'):
    os.makedirs('./images/decision_trees')
    
# change working directory again to where the df.dot file will be made
# set the working directory to be within the newly created ./images/decision_trees
# os.chdir("./images/decision_trees")    
os.chdir("./images/decision_trees")

import graphviz
from sklearn import tree
# You can visualize the trained Decision Tree by first using the export_graphviz()
# method to output a graph definition file called iris_tree.dot

################################# Visualization of Decision Tree ##########################

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42) # Be aware of default parameters
tree_clf.fit(X, y)

export_graphviz(
        tree_clf,
        out_file=image_path("tree.dot"),
        feature_names=feature,
        class_names='churn',
        rounded=True,
        filled=True
    )

# Then you can convert this .dot file to a variety of formats such as PDF or PNG using
# the dot command-line tool from the graphviz package.

# change working directory again to where the iris_tree.dot file will be made
# set the working directory to be within the newly created ./images/decision_trees
os.chdir("./images/decision_trees")

#An alternative way to do it in python
import pydot  # run 'pip install pydot' in anaconda prompt from 'https://stackoverflow.com/questions/53773344/importerrorno-module-named-pydot'
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

# Note: Graphviz is an open source graph visualization software package, available at http://www.graphviz.org/
# Converting .dot file to PNG Example: Run command "dot -Tpng tree.dot -o tree.png" in the terminal after installing graphviz package 
# and making sure you are in the right directory (same directory as the .dot file)

print(os.getcwd())
path_png = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, "tree.png")
Image(filename="tree.png")

# # Model Evaluation

######################################### Imports #########################################
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

###################################### Split the Data ######################################
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=21)
# Note: Always a good idea to shuffle the dataset before you split it into training and testing
# train_test_split performs shuffling by default
# In this case, I made a split of 80/20 between training data and test data

############################# Build Model & Apply it to the Test Set #######################

#Build the decision tree (Per Instruction)
clf1 = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, max_leaf_nodes=14, min_impurity_decrease=0.001, random_state=42) 

# "clf3.fit(X_train, y_train)"" fits the model and then
# ".predict(X_test)" makes predicitions based on the test set
y_pred = clf1.fit(X_train, y_train).predict(X_test)

print(classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn']))

#Build the decision tree (Modified 1)
clf2 = tree.DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=21)

# "clf3.fit(X_train, y_train)"" fits the model and then
# ".predict(X_test)" makes predicitions based on the test set
y_pred_1 = clf2.fit(X_train, y_train).predict(X_test)



###################################### Confusion Matrix #####################################
# Function that prints and plots the confusion matrix.
# Normalization can be applied by setting `normalize=True` (see below for examples)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix by Sean Jung")
    else:
        print('Confusion matrix, without normalization by Sean Jung')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix to evaluate the accuracy of a classification
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# This is a non-normalized confusion matrix of the model
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=('Not terminate', "Terminate"),
                      title='Confusion matrix, without normalization by Sean Jung')

# Classification report (Default)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# This is a normalized confusion matrix of the model (Default)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=('Not terminate', "Terminate"), normalize=True,
                      title='Normalized confusion matrix by Sean Jung')

plt.show()

# Compute confusion matrix to evaluate the accuracy of a classification (Modified 2)
#cnf_matrix_2 = confusion_matrix(y_test, y_pred_2)
#np.set_printoptions(precision=2)

# Classification report (Modified 1)
# print(classification_report(y_test, y_pred_1))