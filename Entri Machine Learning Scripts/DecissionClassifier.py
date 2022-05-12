# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:08:47 2021

@author: Taroon Reddy
"""

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

os.getcwd()

os.chdir('D:\Artificial Intelligence\Datasets')

print(os.getcwd())

df = pd.read_csv('diabetes.csv')

df.shape

df.columns

df.describe().T


df.info()

array = df.values

array

df

----

"Selecting  independent and dependenct variables"


X = array[:,0:8] # ivs for train X


y = array[:,8] # dv y


" Data Partioning"

from sklearn.model_selection import train_test_split 


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30) 

print('Partitioning Done!')

#"Data Modelling for Decission Tree"
 

#Model 1

from sklearn.tree import DecisionTreeClassifier 

from sklearn import metrics



model = DecisionTreeClassifier()


model.fit(X_train,y_train) # 536  #230


prediction = model.predict(X_test) 

prediction


outcome = y_test 

outcome

#from sklearn.tree import export_text



print(metrics.accuracy_score(outcome,prediction))


print(metrics.confusion_matrix(y_test,prediction)) 


from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error 

print(classification_report(y_test, prediction)) 


model.get_depth()

model.get_n_leaves()

model.get_params()

model.score(X,y)




model.feature_importances_

#New Data --- Validation Data or Cross Vali

#Model 2  ---- max_depth = 10   ----73% accuracy



Visulizations:
    ---





text_representation = export_text(model)
print(text_representation)

with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)


from sklearn import tree

df =  pd.read_csv('diabetes.csv')

model = DecisionTreeClassifier()
model.fit(X_train,y_train)# 536  #230


clf = DecisionTreeClassifier(random_state=0)

clf.fit(X_train,y_train)

tree.plot_tree(clf)



from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(X, y)

text_representation = tree.export_text(clf)
print(text_representation)


with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                   feature_names=iris.feature_names,  
                   class_names=iris.target_names,
                   filled=True)

fig.savefig("decistion_tree.png")


#Visualize Decision Tree with graphviz

pip install graphviz


import graphviz
# DOT data
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names,
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph

graph.render("decision_tree_graphivz")

##'decision_tree_graphivz.png'


Plot Decision Tree with dtreeviz Package
----
pip install dtreeviz

from dtreeviz.trees import dtreeviz # remember to load the package

viz = dtreeviz(clf, X, y,
                target_name="target",
                feature_names=iris.feature_names,
                class_names=list(iris.target_names))

viz


---------


Visualizing the Decision Tree in Regression Task
---


from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree


boston = datasets.load_boston()
X = boston.data
y = boston.target


# Fit the regressor, set max_depth = 3
regr = DecisionTreeRegressor(max_depth=3, random_state=1234)
model = regr.fit(X, y)


text_representation = tree.export_text(regr)
print(text_representation)


fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(regr, feature_names=boston.feature_names, filled=True)



dot_data = tree.export_graphviz(regr, out_file=None, 
                                feature_names=boston.feature_names,  
                                filled=True)
graphviz.Source(dot_data, format="png") 





