from sklearn import tree;
from sklearn.datasets import load_iris;
from sklearn.metrics import accuracy_score;
from sklearn.model_selection import train_test_split;
from sklearn.neighbors import KNeighborsClassifier;



def CalculateDecisionTreeAccuracy():
	iris = load_iris();

	data = iris.data;
	target = iris.target;

	data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.25);

	classifier = tree.DecisionTreeClassifier();
	classifier.fit(data_train,target_train);
	predictions = classifier.predict(data_test);

	accuracy = accuracy_score(target_test,predictions);
	return  accuracy;

def CalculateKNNAccuracy():
	iris = load_iris();

	data = iris.data;
	target = iris.target;

	data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.25);

	classifier = KNeighborsClassifier(n_neighbors=5);
	classifier.fit(data_train,target_train);
	predictions = classifier.predict(data_test);

	accuracy = accuracy_score(target_test,predictions);
	return  accuracy;


def main():
	accuracy = CalculateDecisionTreeAccuracy();
	print("Accuracy using Decision Tree algorithm is::",accuracy *100,"%");
	accuracy = CalculateKNNAccuracy();
	print("Accuracy using K-Nearest Neighbour algorithm is::",accuracy *100,"%");



if __name__ == '__main__':
	main()
