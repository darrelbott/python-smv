# Support vector machines (SVM)
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer() #load in data

#print(cancer.feature_names) #check data attributes
#print(cancer.target_names) #check data targets

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
#print(x_train, y_train) #check

classes = ['malignant', 'benign']

#run for loop to get best model
#clf = svm.SVC(kernel="linear", C=2) #linear (C=2), poly (degree=2)
clf = KNeighborsClassifier(n_neighbors=11)
clf.fit(x_train, y_train)

#predict
y_pred = clf.predict(x_test)

#accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)