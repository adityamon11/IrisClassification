from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris();
print("Features of the iris data set",iris.feature_names)
print("Targets of the iris data set",iris.target_names)

#Indices of removed elements
test_index =[1,51,101];

#Training data with removed elements
train_target = np.delete(iris.target,test_index)
train_data = np.delete(iris.data,test_index,axis =0) #axis 0 = first axis; axis runs downwards from the rows

#Testing data
test_target = iris.target[test_index];
test_data = iris.data[test_index]

#Form Decision Tree Classifier
classifier = tree.DecisionTreeClassifier();
classifier.fit(train_data,train_target);

print("Result of testing")
classifier.predict(test_data)

#Visualization
from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO();
tree.export_graphviz(classifier,out_file=dot_data,feature_names=iris.feature_names,class_names=iris.target_names,filled=True
	,rounded=True,impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("IrisDTreeVisual.pdf")